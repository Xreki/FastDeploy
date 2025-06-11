// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cutlass_kernels/moe_gemm/wintx_unzip_impl.h"
#include "helper.h"

template <paddle::DataType T>
void WintxUnzipKernel(const paddle::Tensor &zipped_weight,
                      const paddle::optional<paddle::Tensor> &local_scale,
                      const paddle::optional<paddle::Tensor> &code_scale,
                      const paddle::optional<paddle::Tensor> &code_zip,
                      const paddle::optional<paddle::Tensor> &super_scale,
                      paddle::Tensor &weight, const std::string &quant_method) {
  using DataType_ = typename PDTraits<T>::DataType;
  using data_t = typename PDTraits<T>::data_t;
  using NvType = typename PDTraits<T>::DataType;

  paddle::Tensor *super_scale_tensor =
      const_cast<paddle::Tensor *>(super_scale.get_ptr());
  const auto *super_scale_ptr =
      super_scale_tensor ? super_scale_tensor->data<data_t>() : nullptr;

  auto *weight_ptr = weight.data<data_t>();

  const int64_t batch = weight.shape()[0];
  const int64_t num_rows = weight.shape()[1];
  const int64_t num_columns = weight.shape()[2];

  if (quant_method == "weight_only_int2.5") {
    const auto *zipped_weight_ptr = zipped_weight.data<int16_t>();
    Wint25UnzipKernelLauncher<DataType_>(
        reinterpret_cast<const uint16_t *>(zipped_weight_ptr),
        reinterpret_cast<const NvType *>(super_scale_ptr),
        reinterpret_cast<NvType *>(weight_ptr), batch, num_rows, num_columns);
  } else if (quant_method == "weight_only_int2") {
    paddle::Tensor *local_scale_tensor =
        const_cast<paddle::Tensor *>(local_scale.get_ptr());
    paddle::Tensor *code_scale_tensor =
        const_cast<paddle::Tensor *>(code_scale.get_ptr());
    paddle::Tensor *code_zip_tensor =
        const_cast<paddle::Tensor *>(code_zip.get_ptr());

    Wint2UnzipKernelLauncher<DataType_>(
        zipped_weight.data<uint8_t>(),
        reinterpret_cast<const NvType *>(local_scale_tensor->data<data_t>()),
        code_scale_tensor->data<float>(), code_zip_tensor->data<float>(),
        reinterpret_cast<const NvType *>(super_scale_ptr),
        reinterpret_cast<NvType *>(weight_ptr), batch, num_rows, num_columns);
  } else {
    PD_THROW("Unsupported quant_method for WintxUnzip.");
  }
}

std::vector<paddle::Tensor>
WintXUnzip(const paddle::Tensor &zipped_weight,
           const paddle::optional<paddle::Tensor> &local_scale,
           const paddle::optional<paddle::Tensor> &code_scale,
           const paddle::optional<paddle::Tensor> &code_zip,
           const paddle::optional<paddle::Tensor> &super_scale,
           const std::string &quant_method) {
  paddle::Tensor *local_scale_tensor =
      const_cast<paddle::Tensor *>(local_scale.get_ptr());
  paddle::Tensor *super_scale_tensor =
      const_cast<paddle::Tensor *>(super_scale.get_ptr());
  if (quant_method == "weight_only_int2.5") {
    PD_CHECK(super_scale_tensor, "super_scale must be set in wint2.5!");
  } else if (quant_method == "weight_only_int2") {
    PD_CHECK(local_scale_tensor, "local_scale must be set in wint2.0!");
  }

  auto place = zipped_weight.place();
  auto dtype = super_scale_tensor ? super_scale_tensor->dtype()
                                  : local_scale_tensor->dtype();

  auto output_dims = zipped_weight.dims();
  const int unzip_axis = 1;
  if (quant_method == "weight_only_int2.5") {
    output_dims[unzip_axis] = output_dims[unzip_axis] / 10 * 64;
  } else if (quant_method == "weight_only_int2") {
    output_dims[unzip_axis] = output_dims[unzip_axis] * 4;
  } else {
    PD_THROW("Unsupported data type for WintxUnzip");
  }
  auto output_tensor = GetEmptyTensor(output_dims, dtype, place);

  switch (dtype) {
  case paddle::DataType::BFLOAT16:
    WintxUnzipKernel<paddle::DataType::BFLOAT16>(
        zipped_weight, local_scale, code_scale, code_zip, super_scale,
        output_tensor, quant_method);
    break;
  case paddle::DataType::FLOAT16:
    WintxUnzipKernel<paddle::DataType::FLOAT16>(
        zipped_weight, local_scale, code_scale, code_zip, super_scale,
        output_tensor, quant_method);
    break;
  default:
    PD_THROW("Unsupported data type for WintxUnzip");
  }
  return {output_tensor};
}

std::vector<std::vector<int64_t>> WintXUnzipInferShape(
    const std::vector<int64_t> &zipped_weight_shape,
    const paddle::optional<std::vector<int64_t>> &local_scale_shape,
    const paddle::optional<std::vector<int64_t>> &code_scale_shape,
    const paddle::optional<std::vector<int64_t>> &code_zip_shape,
    const paddle::optional<std::vector<int64_t>> &super_scale_shape,
    const std::string &quant_method) {
  std::vector<int64_t> output_shape(zipped_weight_shape);
  const int unzip_axis = 1;
  if (quant_method == "weight_only_int2.5") {
    output_shape[unzip_axis] = zipped_weight_shape[unzip_axis] / 10 * 64;
    PD_CHECK(output_shape[unzip_axis] % 64 == 0,
             "unzip_size must be divisible by 64 in wint2.5!");
  } else if (quant_method == "weight_only_int2") {
    output_shape[unzip_axis] = zipped_weight_shape[unzip_axis] * 4;
    PD_CHECK(output_shape[unzip_axis] % 64 == 0,
             "unzip_size must be divisible by 64 in wint2!");
  } else {
    PD_THROW("Unsupported quant_type for WintxUnzip");
  }
  return {output_shape};
}

std::vector<paddle::DataType> WintXUnzipInferDtype(
    const paddle::DataType &zipped_weight_dtype,
    const paddle::optional<paddle::DataType> &local_scale_dtype,
    const paddle::optional<paddle::DataType> &code_scale_dtype,
    const paddle::optional<paddle::DataType> &code_zip_dtype,
    const paddle::optional<paddle::DataType> &super_scale_dtype,
    const std::string &quant_method) {
  if (quant_method == "weight_only_int2.5") {
    return {super_scale_dtype.get()};
  } else if (quant_method == "weight_only_int2") {
    return {local_scale_dtype.get()};
  } else {
    PD_THROW("Unsupported quant_type for WintxUnzip");
  }
}

PD_BUILD_STATIC_OP(winx_unzip)
    .Inputs({"zipped_weight", paddle::Optional("local_scale"),
             paddle::Optional("code_scale"), paddle::Optional("zip_scale"),
             paddle::Optional("super_scale")})
    .Outputs({"weight"})
    .Attrs({"quant_method:std::string"})
    .SetKernelFn(PD_KERNEL(WintXUnzip))
    .SetInferShapeFn(PD_INFER_SHAPE(WintXUnzipInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WintXUnzipInferDtype));
