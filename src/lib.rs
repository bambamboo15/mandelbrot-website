mod floatexp;
mod mandelbrot;
use mandelbrot::MandelbrotEngine;
use wasm_bindgen::prelude::*;

use crate::floatexp::Float;

#[wasm_bindgen]
struct MandelbrotApp {
    engine: MandelbrotEngine,
    width: f32,
    height: f32,
}

#[wasm_bindgen]
impl MandelbrotApp {
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Self {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        let mut engine = MandelbrotEngine::new(
            wgpu::SurfaceTarget::Canvas(canvas.clone()),
            canvas.width(),
            canvas.height(),
        )
        .await;

        engine.pan = [Float::try_from(-0.75).unwrap(), Float::ZERO];
        engine.zoom = 0.0;

        MandelbrotApp {
            engine,
            width: 0.0,
            height: 0.0,
        }
    }

    pub fn tick(
        &mut self,
        delta_x: f32,
        delta_y: f32,
        zoom_delta: f32,
        cursor_x: f32,
        cursor_y: f32,
    ) {
        let mut is_dirty = false;

        if delta_x != 0.0 || delta_y != 0.0 {
            is_dirty = true;

            let (delta_real, delta_imag) = self.engine.complex_from_pixel_offset(delta_x, delta_y);
            self.engine.pan[0] -= delta_real;
            self.engine.pan[1] += delta_imag;
        }

        if zoom_delta != 0.0 {
            is_dirty = true;

            let dx = cursor_x - self.width * 0.5;
            let dy = cursor_y - self.height * 0.5;

            let first = self.engine.complex_from_pixel_offset(dx, dy);
            self.engine.zoom = f32::max(self.engine.zoom - 0.5 * zoom_delta, 0.0);
            let second = self.engine.complex_from_pixel_offset(dx, dy);

            self.engine.pan[0] -= second.0 - first.0;
            self.engine.pan[1] += second.1 - first.1;
        }

        if is_dirty {
            self.engine.pan = std::mem::take(&mut self.engine.pan)
                .map(|x| x.clamp(Float::from(-2), Float::from(2)));
            self.engine.update();
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.engine.resize(width, height);
        self.width = width as f32;
        self.height = height as f32;
    }
}
