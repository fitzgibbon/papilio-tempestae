#[cfg(target_arch = "never_spirv")]
macro_rules! unsupported_features {
    ($($feature:literal),+ $(,)?) => {
        $(
            #[cfg(feature = $feature)]
            compile_error!(
                concat!(
                    "`",
                    $feature,
                    "`",
                    " feature is not supported when building for SPIR-V.",
                )
            );
        )+
    }
}

#[cfg(target_arch = "never_spirv")]
unsupported_features! {
    "approx",
    "debug-glam-assert",
    "glam-assert",
    "rand",
    "serde",
    "std",
}
