mod floatexp;
mod mandelbrot;
use crate::floatexp::Float;
use dashu::float::{FBig, round::mode::Zero};
use mandelbrot::MandelbrotEngine;
use std::str::FromStr;
use wasm_bindgen::prelude::*;
use web_sys::js_sys;

type Decimal = FBig<Zero, 10>;

#[wasm_bindgen]
struct MandelbrotApp {
    engine: MandelbrotEngine,
    width: f32,
    height: f32,
    on_update: js_sys::Function,
}

#[wasm_bindgen]
impl MandelbrotApp {
    pub async fn create(canvas: web_sys::HtmlCanvasElement, on_update: js_sys::Function) -> Self {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        let engine = MandelbrotEngine::new(
            wgpu::SurfaceTarget::Canvas(canvas.clone()),
            canvas.width(),
            canvas.height(),
        )
        .await;

        let mut app = MandelbrotApp {
            engine,
            width: 0.0,
            height: 0.0,
            on_update,
        };
        app.apply("-0.75", "0.0", 0.0, 400, 2);
        app
    }

    pub fn tick(
        &mut self,
        delta_x: f32,
        delta_y: f32,
        zoom_delta: f32,
        cursor_x: f32,
        cursor_y: f32,
    ) {
        if delta_x != 0.0 || delta_y != 0.0 {
            let (delta_real, delta_imag) = self.engine.complex_from_pixel_offset(delta_x, delta_y);
            self.engine.pan_by([-delta_real, delta_imag]);
        }

        if zoom_delta != 0.0 {
            let dx = cursor_x - self.width * 0.5;
            let dy = cursor_y - self.height * 0.5;

            let first = self.engine.complex_from_pixel_offset(dx, dy);
            self.engine.set_zoom(self.engine.zoom() - 0.5 * zoom_delta);
            let second = self.engine.complex_from_pixel_offset(dx, dy);
            self.engine.pan_by([first.0 - second.0, second.1 - first.1]);
        }

        if self.engine.tick() {
            self.on_update();
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.engine.resize(width, height);
        self.width = width as f32;
        self.height = height as f32;
    }

    pub fn apply(&mut self, real: &str, imag: &str, zoom: f32, iterations: usize, pixelation: u8) {
        self.engine.set_pan([real, imag].map(|x| {
            Decimal::from_str(x)
                .unwrap()
                .to_binary()
                .value()
                .clamp(Float::from(-2), Float::from(2))
        }));
        self.engine.set_zoom(zoom);
        self.engine.set_iterations(iterations);
        self.engine.set_pixelation(pixelation);
    }

    fn on_update(&self) {
        let [real, imag] = (self.engine.pan())
            .clone()
            .map(|x| x.to_decimal().value().to_string());
        let _ = self.on_update.call(
            &JsValue::null(),
            (
                &JsValue::from(real),
                &JsValue::from(imag),
                &JsValue::from(self.engine.zoom()),
                &JsValue::from(self.engine.iterations()),
                &JsValue::from(self.engine.pixelation()),
            ),
        );
    }
}
