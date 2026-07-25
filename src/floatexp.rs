use dashu::{
    base::{BitTest, Sign},
    float::{FBig, Word, round::mode::Zero},
};

/// Arbitrary-precision floating-point type [`FBig<Zero, 2>`].
pub type Float = FBig<Zero, 2>;

/// Floating-point type that has an arbitrary exponent but only `f32` precision.
///
/// This is internally stored as a [0.5, 1.0) mantissa and corresponding exponent.
#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FloatExp {
    mantissa: f32,
    exponent: i32,
}

impl FloatExp {
    /// Constructs a [`FloatExp`] from a floating-point exponent value.
    /// Conceptually this will hold `2.0` raised to the power of the parameter.
    #[must_use]
    pub fn from_exponent(exponent: f32) -> Option<Self> {
        if (exponent.is_nan() || exponent.is_infinite())
            || (exponent < i32::MIN as f32 || exponent > i32::MAX as f32)
        {
            return None;
        }

        let exponent_whole = exponent.floor() as i32;
        let exponent_frac = exponent - exponent_whole as f32;

        // Obviously we know that `exponent_frac` is in the range [0.0, 1.0) so the
        // following operation will push it into [1.0, 2.0) which is what we want.
        let mantissa = 2.0f32.powf(exponent_frac);
        let exponent = exponent_whole;

        Some(FloatExp {
            mantissa: mantissa * 0.5,
            exponent: exponent + 1,
        })
    }

    /// Constructs a [`FloatExp`] from an unnormalized mantissa and exponent.
    #[must_use]
    pub fn from_parts(mantissa: f32, exponent: i32) -> Self {
        let (res_mantissa, res_exponent) = libm::frexpf(mantissa);
        FloatExp {
            mantissa: res_mantissa,
            exponent: res_exponent + exponent,
        }
    }

    /// Returns the (-1.0, -0.5] ∪ [0.5, 1.0) mantissa.
    #[must_use]
    pub fn mantissa(self) -> f32 {
        self.mantissa
    }

    /// Returns the exponent.
    #[must_use]
    pub fn exponent(self) -> i32 {
        self.exponent
    }

    /// Performs the square operation.
    #[must_use]
    pub fn sqr(mut self) -> Self {
        self.mantissa *= self.mantissa;
        self.exponent += self.exponent;
        if self.mantissa < 0.5 {
            self.mantissa += self.mantissa;
            self.exponent -= 1;
        }
        self
    }

    /// Performs the square root operation.
    #[must_use]
    pub fn sqrt(mut self) -> Self {
        self.mantissa = self.mantissa.sqrt();
        if self.exponent & 1 != 0 {
            self.mantissa *= if self.exponent > 0 {
                std::f32::consts::SQRT_2
            } else {
                std::f32::consts::FRAC_1_SQRT_2
            };
        }
        self.exponent >>= 1;
        self
    }
}

impl std::ops::Add<FloatExp> for FloatExp {
    type Output = FloatExp;

    fn add(self, rhs: FloatExp) -> Self::Output {
        // EDGE CASE: When adding a number with 0.0, obviously that should keep the number itself.
        // However, when that 0.0 had a large exponent, the number was often scaled down as it had
        // a smaller exponent, underflowing it. Thus we make 0.0 conceptually have an exponent of zero.
        if self.mantissa == 0.0 {
            return rhs;
        }
        if rhs.mantissa == 0.0 {
            return self;
        }

        let (small, big) = if self.exponent <= rhs.exponent {
            (self, rhs)
        } else {
            (rhs, self)
        };

        let combined_mantissa =
            big.mantissa + libm::ldexpf(small.mantissa, small.exponent - big.exponent);
        return FloatExp::from_parts(combined_mantissa, big.exponent);
    }
}

impl std::ops::Sub<FloatExp> for FloatExp {
    type Output = FloatExp;

    fn sub(self, rhs: FloatExp) -> Self::Output {
        self + FloatExp {
            mantissa: -rhs.mantissa,
            exponent: rhs.exponent,
        }
    }
}

impl std::ops::Mul<FloatExp> for FloatExp {
    type Output = FloatExp;

    fn mul(self, rhs: FloatExp) -> Self::Output {
        let mut mantissa = self.mantissa * rhs.mantissa;
        let mut exponent = self.exponent + rhs.exponent;
        if mantissa.abs() < 0.5 {
            mantissa += mantissa;
            exponent -= 1;
        }
        FloatExp { mantissa, exponent }
    }
}

impl From<f32> for FloatExp {
    fn from(value: f32) -> Self {
        Self::from_parts(value, 0)
    }
}

impl TryFrom<&Float> for FloatExp {
    type Error = ();

    fn try_from(value: &Float) -> Result<Self, Self::Error> {
        let repr = value.repr();
        let significand = repr.significand();
        let exponent = repr.exponent();

        if repr.is_infinite() {
            return Err(());
        }
        if significand.is_zero() {
            return Ok(Self {
                mantissa: 0.0,
                exponent: 0,
            });
        }

        // When we are deeply zoomed into the Mandelbrot set, the integer `significand` is going to be
        // impossibly long. For our `f32` mantissa, we only need the top 24 bits (`mantissa_bits`),
        // then discard the topmost one, leaving 23.
        let num_bits = significand.bit_len();
        let (sign, words) = significand.as_sign_words();

        let mut remaining = num_bits % Word::BITS as usize;
        if remaining == 0 {
            remaining = Word::BITS as usize;
        }
        let mantissa_bits: u32 = if remaining < 24 {
            // 00000000001XXXXXXXXXXXXXXXXXXXXX XX..............................
            // 000000001XXXXXXXXXXXXXXXXXXXXXXX ................................
            let top = words[words.len() - 1];
            let second = words.get(words.len() - 2).copied().unwrap_or(0);
            let transfer = 24 - remaining;
            ((top << transfer) | (second >> (Word::BITS as usize - transfer))) as u32
        } else {
            // 000001XXXXXXXXXXXXXXXXXXXXXXX000
            // 000000001XXXXXXXXXXXXXXXXXXXXXXX
            (words.last().unwrap() >> (remaining - 24)) as u32
        };

        // Reconstruct the mantissa and exponent.
        let reference: f32 = match sign {
            Sign::Positive => 1.0,
            Sign::Negative => -1.0,
        };
        let mantissa: f32 = f32::from_bits(f32::to_bits(reference) | (mantissa_bits & 0x7FFFFF));
        let exponent: i32 = (exponent + (num_bits as isize - 1)).try_into().unwrap();
        Ok(FloatExp {
            mantissa: mantissa * 0.5,
            exponent: exponent + 1,
        })
    }
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComplexExp {
    pub real: FloatExp,
    pub imag: FloatExp,
}

impl ComplexExp {
    #[must_use]
    pub fn new(real: FloatExp, imag: FloatExp) -> Self {
        ComplexExp { real, imag }
    }

    #[must_use]
    pub fn from_floats(real: &Float, imag: &Float) -> Option<Self> {
        Some(Self::new(real.try_into().ok()?, imag.try_into().ok()?))
    }

    #[must_use]
    pub fn length_squared(self) -> FloatExp {
        self.real.sqr() + self.imag.sqr()
    }

    #[must_use]
    pub fn length(self) -> FloatExp {
        self.length_squared().sqrt()
    }
}

impl std::ops::Add<ComplexExp> for ComplexExp {
    type Output = ComplexExp;

    fn add(self, rhs: ComplexExp) -> Self::Output {
        Self::new(self.real + rhs.real, self.imag + rhs.imag)
    }
}

impl std::ops::Sub<ComplexExp> for ComplexExp {
    type Output = ComplexExp;

    fn sub(self, rhs: ComplexExp) -> Self::Output {
        Self::new(self.real - rhs.real, self.imag - rhs.imag)
    }
}

impl std::ops::Mul<ComplexExp> for ComplexExp {
    type Output = ComplexExp;

    fn mul(self, rhs: ComplexExp) -> Self::Output {
        Self::new(
            self.real * rhs.real - self.imag * rhs.imag,
            self.real * rhs.imag + self.imag * rhs.real,
        )
    }
}

impl From<f32> for ComplexExp {
    fn from(value: f32) -> Self {
        Self::from(FloatExp::from(value))
    }
}

impl From<FloatExp> for ComplexExp {
    fn from(value: FloatExp) -> Self {
        Self::new(value, FloatExp::from(0.0))
    }
}
