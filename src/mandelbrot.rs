use crate::floatexp::*;

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MandelbrotUniforms {
    max_ref_iteration: i32,
    max_iteration: i32,
    mag: FloatExp,
    res: [f32; 2],
    _padding: [f32; 2],
}

/// Renders Mandelbrot with a GPU.
pub struct MandelbrotEngine {
    // Core WGPU variables.
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface<'static>,

    // Data storage buffers.
    uniform_buffer: wgpu::Buffer,
    orbit_storage_buffer: wgpu::Buffer,
    uniforms: MandelbrotUniforms,
    orbit: Vec<ComplexExp>,

    // Pipeline 1: the compute math engine.
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    compute_bind_group_layout: wgpu::BindGroupLayout,

    // Pipeline 2: the presentation blit/upscale engine.
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    bilinear_sampler: wgpu::Sampler,

    // Intermediate canvas assets.
    shared_texture: wgpu::Texture,
    shared_texture_view: wgpu::TextureView,

    // Mathematical configuration.
    pub iterations: usize,
    pub pan: [Float; 2],
    /// Exponential zoom. For instance, `0.0` is identity, `1.0` means zoomed by a factor of two, etc.
    /// Higher zoom values make for significantly smaller steps across pixels.
    pub zoom: f32,
}

impl MandelbrotEngine {
    pub async fn new(target: wgpu::SurfaceTarget<'static>, width: u32, height: u32) -> Self {
        // Initialize WGPU instance to target the surface.
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(target).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        // Configure the surface layout using the dimensions of the surface.
        let config = surface.get_default_config(&adapter, width, height).unwrap();
        surface.configure(&device, &config);

        // Create the uniform buffer (storage for constant values that the shader processes).
        // Additionally, create the storager buffer which holds perturbation orbits.
        // For now, we cap the number of iterations.
        let iterations = 1000;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Controls Uniform Buffer"),
            size: std::mem::size_of::<MandelbrotUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let orbit_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Orbit Storage Buffer"),
            size: (iterations * std::mem::size_of::<ComplexExp>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize the shared canvas assets.
        let shared_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shared Compute Canvas Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shared_texture_view =
            shared_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bilinear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bilinear Upscale Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Initialize the compute pipeline layout and bind group.
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Layout Template"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: orbit_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shared_texture_view),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Math Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    ..Default::default()
                }),
            ),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Initialize blit pipeline layout and bind group.
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit Presentation Layout Template"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group"),
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shared_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&bilinear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Render Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[Some(&blit_bind_group_layout)],
                    ..Default::default()
                }),
            ),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            device,
            queue,
            config,
            surface,
            uniform_buffer,
            orbit_storage_buffer,
            uniforms: Default::default(),
            orbit: Vec::new(),
            compute_pipeline,
            compute_bind_group,
            compute_bind_group_layout,
            blit_pipeline,
            blit_bind_group,
            blit_bind_group_layout,
            bilinear_sampler,
            shared_texture,
            shared_texture_view,
            iterations,
            pan: [Float::ZERO, Float::ZERO],
            zoom: 0.0,
        }
    }

    /// Helper function that computes the complex number offset from a pixel offset.
    ///
    /// This does not care about the orientation of pixels: a positive Y pixel offset
    /// will result in a positive output.
    pub fn complex_from_pixel_offset(&self, delta_x: f32, delta_y: f32) -> (Float, Float) {
        let mag = -self.zoom;
        let mag_floor = mag.floor();
        let mag_int = mag_floor as i32;
        let mag_frac = mag - mag_floor;

        let pixels = [delta_x, delta_y];
        let res = self.uniforms.res;

        let [complex_x, complex_y] = [0, 1].map(|i| {
            let aspect_ratio = f32::max(res[0], res[1]) / res[1 - i];
            let unscaled = 0.665 * aspect_ratio * (pixels[i] / res[i]) * 2.0_f32.powf(mag_frac);
            Float::try_from(unscaled).unwrap() * Float::from(2).powf(&Float::from(mag_int))
        });
        (complex_x, complex_y)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        // Reconfigure the main surface.
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);

        // Destroy the old canvas and reallocate the texture at the new native size.
        self.shared_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shared Compute Canvas Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.shared_texture_view = self
            .shared_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Rebuild "Compute Bind Group" using the saved template layout.
        self.compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group (Resize Update)"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.orbit_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.shared_texture_view),
                },
            ],
        });

        // Rebuild "Blit Bind Group" using the saved template layout and sampler.
        self.blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group (Resize Update)"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.shared_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.bilinear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        self.update();
    }

    /// Calculates and draws a new render.
    pub fn update(&mut self) {
        self.calculate();
        self.draw();
    }

    fn calculate(&mut self) {
        // Adjust the precision of `pan` so that the reference orbit calculates correctly.
        // I don't know what much else to do other than the heuristic of adding 64 to the zoom level.
        self.pan = std::mem::take(&mut self.pan)
            .map(|x| x.with_precision(self.zoom as usize + 64).value());

        // Compute the orbit buffer on the CPU in full multiprecision. Thanks to perturbation and
        // excellent rebasing algorithms we only need to ever do this once for the center point.
        // The reference orbit does not have to take up maximum iterations.
        self.orbit = {
            let mut orbit: Vec<ComplexExp> = Vec::new();
            let (c, mut z) = (self.pan.clone(), [Float::ZERO, Float::ZERO]);
            for _ in 0..self.iterations {
                if z[0].sqr() + z[1].sqr() > Float::from(64) {
                    break;
                }
                orbit.push(ComplexExp {
                    x: (&z[0]).try_into().unwrap(),
                    y: (&z[1]).try_into().unwrap(),
                });
                z = [
                    z[0].sqr() - z[1].sqr() + &c[0],
                    Float::from(2) * &z[0] * &z[1] + &c[1],
                ];
            }
            orbit
        };

        self.uniforms.max_ref_iteration = (self.orbit.len() as i32) - 1;
        self.uniforms.max_iteration = self.iterations as i32;
        self.uniforms.mag = FloatExp::from_exponent(-self.zoom).unwrap();
    }

    fn draw(&mut self) {
        // If the current number of iterations (upper bound for CPU orbit buffer length) is larger
        // than the length of the GPU orbit buffer, we have to destroy and reallocate the GPU buffer
        // accordingly.
        let max_cpu_orbit_len = self.iterations;
        let gpu_orbit_len =
            self.orbit_storage_buffer.size() as usize / std::mem::size_of::<ComplexExp>();
        if max_cpu_orbit_len > gpu_orbit_len {
            let raw_size = (max_cpu_orbit_len * std::mem::size_of::<ComplexExp>()) as u64;
            let aligned_size = (raw_size + 31) & !31;

            self.orbit_storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Orbit Storage Buffer"),
                size: aligned_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group"),
                layout: &self.compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.orbit_storage_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.shared_texture_view),
                    },
                ],
            });
        }

        // Populate all buffers.
        let scale_factor = 1.0;
        self.uniforms.res = [
            self.config.width as f32 * scale_factor,
            self.config.height as f32 * scale_factor,
        ];
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
        self.queue.write_buffer(
            &self.orbit_storage_buffer,
            0,
            bytemuck::cast_slice(&self.orbit),
        );

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            _ => return,
        };
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // Pass 1: execute the compute shader thread grid.
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

            let workgroup_count_x = (self.config.width + 15) / 16;
            let workgroup_count_y = (self.config.height + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        // Pass 2: draw the compute output texture to screen 1:1 via the blit pipeline.
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Presentation Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.blit_pipeline);
            render_pass.set_bind_group(0, &self.blit_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(surface_texture);
    }
}
