//! A thin wrapper around [sys] providing [Result]s with [CudartError].

use super::sys;
use core::ffi::{c_uchar, c_uint, c_void, CStr};
/// Wrapper around [sys::cudaError_t].
/// See nvidia's [cudaError_t docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6)

pub type CudartResult<T> = Result<T, CudartError>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CudartError(pub sys::cudaError_t);

impl sys::cudaError_t {
    /// Transforms into a [Result] of [CudartError]
    pub fn result(self) -> Result<(), CudartError> {
        match self {
            sys::cudaError_t::cudaSuccess => Ok(()),
            _ => Err(CudartError(self)),
        }
    }
}

impl CudartError {
    /// Gets the name for this error.
    ///
    /// See [cudaGetErrorName() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1gb3de7da2f23736878270026dcfc70075)
    pub fn error_name(&self) -> &CStr {
        unsafe {
            let err_str = sys::cudaGetErrorName(self.0);
            CStr::from_ptr(err_str)
        }
    }

    /// Gets the error string for this error.
    ///
    /// See [cudaGetErrorString() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g4bc9e35a618dfd0877c29c8ee45148f1)
    pub fn error_string(&self) -> &CStr {
        unsafe {
            let err_str = sys::cudaGetErrorString(self.0);
            CStr::from_ptr(err_str)
        }
    }
}

impl std::fmt::Debug for CudartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str = self.error_string();
        f.debug_tuple("CudartError")
            .field(&self.0)
            .field(&err_str)
            .finish()
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CudartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CudartError {}

/// Initializes the CUDA runtime API.
/// **Typically Not Required**
///
/// To mitigate any potential confusion, consider this function as a placeholder that provides additional guidance. In the CUDA runtime, 'init' is implicitly implemented, and the initialization process is triggered upon your first call. Therefore, explicit use of this function is typically not required.
/// If you need to initialize a specific device, please refer to [device].
///
/// See [programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization)
///
/// See also [cudaInitDevice() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419)
///
/// For CUDA Driver API Interactions, see [Interactions with the CUDA Driver API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER)
pub fn init() {}

pub mod device {
    //! Device management module
    //!
    //! See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE)

    use super::{sys, CudartError};
    use core::ffi::c_int;
    use std::mem::MaybeUninit;

    /// **CudartDevice is a cudarc alias, defined here in order to clarify the device**
    pub type CudartDevice = ::core::ffi::c_int;

    /// Get the current device for the calling host thread.
    ///
    /// Arguments:
    ///
    /// * `device` - An integer pointer to a device.
    ///
    /// See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78).
    pub fn get() -> Result<CudartDevice, CudartError> {
        let mut dev: CudartDevice = 0;
        unsafe {
            sys::cudaGetDevice(&mut dev).result()?;
            Ok(dev)
        }
    }

    /// Gets the number of available devices.
    /// See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f)
    pub fn get_count() -> Result<c_int, CudartError> {
        let mut count = MaybeUninit::uninit();
        unsafe {
            sys::cudaGetDeviceCount(count.as_mut_ptr()).result()?;
            Ok(count.assume_init())
        }
    }

    /// Get the property of the device.
    ///
    /// See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0)
    ///
    /// # Safety
    /// Must be a device returned from [get].
    pub fn get_property(device: CudartDevice) -> Result<sys::cudaDeviceProp, CudartError> {
        unsafe {
            let mut prop = MaybeUninit::<sys::cudaDeviceProp>::uninit();
            sys::cudaGetDeviceProperties_v2(prop.as_mut_ptr(), device).result()?;
            Ok(prop.assume_init())
        }
    }

    /// Get the total amount of memory in bytes on the device.
    ///
    /// equal to get_property().totalGlobalMem
    ///
    /// See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0)
    /// Also see [cudaDeviceProp]()
    /// # Safety
    /// Must be a device returned from [get].
    pub fn total_mem(device: CudartDevice) -> Result<usize, CudartError> {
        unsafe {
            let mut prop = MaybeUninit::<sys::cudaDeviceProp>::uninit();
            sys::cudaGetDeviceProperties_v2(prop.as_mut_ptr(), device).result()?;
            Ok(prop.assume_init().totalGlobalMem as usize)
        }
    }

    /// Get an attribute of a device.
    ///
    /// See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151)
    ///
    /// # Safety
    /// Must be a device returned from [get].
    pub fn get_attribute(
        device: CudartDevice,
        attribute: sys::cudaDeviceAttr,
    ) -> Result<i32, CudartError> {
        unsafe {
            let mut value = MaybeUninit::uninit();
            sys::cudaDeviceGetAttribute(value.as_mut_ptr(), attribute, device).result()?;
            Ok(value.assume_init())
        }
    }
}

/// for define CUDART_DEVICE in cuda_runtime.h, temporally failed to bind max_potential_block_size and max_potential_block_size_with_flags
///
pub mod occupancy {
    use core::{
        ffi::{c_int, c_uint, c_void},
        mem::MaybeUninit,
    };

    use super::{sys, CudartError};

    /// Returns dynamic shared memory available per block when launching numBlocks blocks on SM.
    ///
    /// See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gc5896a36586821e8bb51d4e837b55bb6)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn available_dynamic_shared_mem_per_block(
        f: sys::cudaFunction_t,
        num_blocks: c_int,
        block_size: c_int,
    ) -> Result<usize, CudartError> {
        let mut dynamic_smem_size = MaybeUninit::uninit();
        unsafe {
            sys::cudaOccupancyAvailableDynamicSMemPerBlock(
                dynamic_smem_size.as_mut_ptr(),
                f as *const c_void,
                num_blocks,
                block_size,
            )
            .result()?;
        }
        Ok(dynamic_smem_size.assume_init())
    }

    /// Returns occupancy of a function.
    ///
    /// See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g5a5d67a3c907371559ba692195e8a38c)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn max_active_block_per_multiprocessor(
        f: sys::cudaFunction_t,
        block_size: c_int,
        dynamic_smem_size: usize,
    ) -> Result<i32, CudartError> {
        let mut num_blocks = MaybeUninit::uninit();
        unsafe {
            sys::cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                num_blocks.as_mut_ptr(),
                f as *const c_void,
                block_size,
                dynamic_smem_size,
            )
            .result()?;
        }
        Ok(num_blocks.assume_init())
    }

    /// Returns occupancy of a function.
    ///
    /// See [cudarc docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g603b86b20b37823253ff89fe8688ba83)
    ///
    /// # Safety
    /// Function must exist. No invalid flags.
    pub unsafe fn max_active_block_per_multiprocessor_with_flags(
        f: sys::cudaFunction_t,
        block_size: c_int,
        dynamic_smem_size: usize,
        flags: c_uint,
    ) -> Result<i32, CudartError> {
        let mut num_blocks = MaybeUninit::uninit();
        unsafe {
            sys::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                num_blocks.as_mut_ptr(),
                f as *const c_void,
                block_size,
                dynamic_smem_size,
                flags,
            )
            .result()?;
        }
        Ok(num_blocks.assume_init())
    }

    // /// Suggest a launch configuration with reasonable occupancy.
    // ///
    // /// Returns (min_grid_size, block_size)
    // ///
    // /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03)
    // ///
    // /// # Safety
    // /// Function must exist and the shared memory function must be correct.  No invalid flags.
    // pub unsafe fn max_potential_block_size(
    //     f: sys::cudaFunction_t,
    //     block_size_to_dynamic_smem_size: sys::CUoccupancyB2DSize,
    //     dynamic_smem_size: usize,
    //     block_size_limit: c_int,
    // ) -> Result<(i32, i32), DriverError> {
    //     let mut min_grid_size = MaybeUninit::uninit();
    //     let mut block_size = MaybeUninit::uninit();
    //     unsafe {
    //         sys::cudaOccupancyMaxPotentialBlockSize(
    //             min_grid_size.as_mut_ptr(),
    //             block_size.as_mut_ptr(),
    //             f,
    //             block_size_to_dynamic_smem_size,
    //             dynamic_smem_size,
    //             block_size_limit,
    //         )
    //         .result()?;
    //     }
    //     Ok((min_grid_size.assume_init(), block_size.assume_init()))
    // }

    // /// Suggest a launch configuration with reasonable occupancy.
    // ///
    // /// Returns (min_grid_size, block_size)
    // ///
    // /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g04c0bb65630f82d9b99a5ca0203ee5aa)
    // ///
    // /// # Safety
    // /// Function must exist and the shared memory function must be correct.  No invalid flags.
    // pub unsafe fn max_potential_block_size_with_flags(
    //     f: sys::cudaFunction_t,
    //     block_size_to_dynamic_smem_size: sys::CUoccupancyB2DSize,
    //     dynamic_smem_size: usize,
    //     block_size_limit: c_int,
    //     flags: c_uint,
    // ) -> Result<(i32, i32), DriverError> {
    //     let mut min_grid_size = MaybeUninit::uninit();
    //     let mut block_size = MaybeUninit::uninit();
    //     unsafe {
    //         sys::cudaOccupancyMaxPotentialBlockSizeWithFlags(
    //             min_grid_size.as_mut_ptr(),
    //             block_size.as_mut_ptr(),
    //             f,
    //             block_size_to_dynamic_smem_size,
    //             dynamic_smem_size,
    //             block_size_limit,
    //             flags,
    //         )
    //         .result()?;
    //     }
    //     Ok((min_grid_size.assume_init(), block_size.assume_init()))
    // }
}
