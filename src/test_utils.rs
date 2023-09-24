macro_rules! function_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        &name[..name.len() - 3]
    }}
}

macro_rules! timeit {
    ($e:expr) => {
        {
            let clk = unsafe { core::arch::x86_64::_rdtsc() };
            let res = $e;
            let endclk = unsafe { core::arch::x86_64::_rdtsc() };
            (endclk - clk, res)
        }
    }
}

pub(crate) use function_name;
pub(crate) use timeit;
