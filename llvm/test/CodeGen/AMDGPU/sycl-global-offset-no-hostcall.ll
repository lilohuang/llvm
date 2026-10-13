; RUN: opt -passes='default<O2>' %s | llc -o - | FileCheck %s --implicit-check-not=hidden_hostcall_buffer
; RUN: opt -passes='lto<O2>' %s | llc -o - | FileCheck %s --implicit-check-not=hidden_hostcall_buffer
;
; Lower SYCL global offsets before inferring implicit kernel arguments. An
; unresolved offset builtin must not make a kernel require PCIe hostcalls.

target triple = "amdgpu11.02-amd-amdhsa"

declare i64 @_Z27__spirv_BuiltInGlobalOffseti(i32)

define amdgpu_kernel void @kernel(ptr addrspace(1) %out) {
  %offset = call i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 0)
  store i64 %offset, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: .name: kernel
; CHECK: .name: kernel_with_offset

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"sycl-device", i32 1}
