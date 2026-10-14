#!/usr/bin/env bash
set -euo pipefail

readonly FOLDSEEK_COMMIT="8dc75c74ad0eddab73cfd905963d13bf74dc012b"

usage() {
  cat <<'EOF'
Usage: scripts/install_foldseek_linux.sh [--gpu | --cpu]

Build the pipeline's pinned Foldseek commit inside the active Conda environment.
GPU is the canonical default. CPU builds are for development and create a
different, non-canonical dataset.

Environment overrides:
  FOLDSEEK_CUDA_ARCHITECTURES  CMake CUDA architecture list (default: native)
  BUILD_JOBS                   Parallel build jobs (default: detected CPU count)
EOF
}

mode="gpu"
case "${1:-}" in
  ""|--gpu) mode="gpu" ;;
  --cpu) mode="cpu" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer supports Linux. On Windows, run scripts/install_foldseek_windows.ps1, which delegates to WSL2." >&2
  exit 2
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the pipeline Conda environment before installing Foldseek." >&2
  exit 2
fi
for tool in git cmake ninja rustc; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required build tool is missing: $tool" >&2
    exit 2
  fi
done

enable_cuda=0
if [[ "$mode" == "gpu" ]]; then
  enable_cuda=1
  for tool in nvcc nvidia-smi; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "GPU prerequisite is missing: $tool" >&2
      exit 2
    fi
  done
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia-smi cannot see a GPU. Check the NVIDIA driver and, on Windows, WSL2 GPU access." >&2
    exit 2
  fi
else
  echo "WARNING: installing the non-canonical CPU Foldseek build." >&2
fi

readonly source_dir="${CONDA_PREFIX}/share/foldseek-${FOLDSEEK_COMMIT}"
readonly build_dir="${CONDA_PREFIX}/share/foldseek-build-${FOLDSEEK_COMMIT}-${mode}"
readonly output_bin="${CONDA_PREFIX}/bin/foldseek-pipeline"

if [[ ! -d "${source_dir}/.git" ]]; then
  mkdir -p "$source_dir"
  git -C "$source_dir" init
  git -C "$source_dir" remote add origin https://github.com/steineggerlab/foldseek.git
fi
git -C "$source_dir" fetch --depth 1 origin "$FOLDSEEK_COMMIT"
git -C "$source_dir" checkout --detach FETCH_HEAD

cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
  -DHAVE_TESTS=0
  -DHAVE_MPI=0
  -DENABLE_CUDA="$enable_cuda"
  -DENABLE_PROSTT5=0
)

# Select the host CPU target independently of CUDA GPU execution.
case "$(uname -m)" in
  x86_64)
    if [[ -r /proc/cpuinfo ]] && grep -Eq '(^|[[:space:]])avx2([[:space:]]|$)' /proc/cpuinfo; then
      cmake_args+=(-DHAVE_AVX2=1)
    else
      cmake_args+=(-DHAVE_SSE4_1=1)
    fi
    ;;
  aarch64)
    cmake_args+=(-DHAVE_ARM8=1)
    ;;
  *)
    echo "Unsupported Linux architecture: $(uname -m)" >&2
    exit 2
    ;;
esac

if [[ "$mode" == "gpu" ]]; then
  readonly cuda_architectures="${FOLDSEEK_CUDA_ARCHITECTURES:-native}"
  readonly nvcc_path="$(command -v nvcc)"
  readonly cuda_root="${CUDAToolkit_ROOT:-$(cd "$(dirname "$nvcc_path")/.." && pwd)}"
  export CUDACXX="$nvcc_path"
  cmake_args+=(
    "-DCUDAToolkit_ROOT=${cuda_root}"
    "-DCMAKE_CUDA_ARCHITECTURES=${cuda_architectures}"
  )
fi

cmake -S "$source_dir" -B "$build_dir" -G Ninja "${cmake_args[@]}"
if ! grep -q "^ENABLE_CUDA:BOOL=${enable_cuda}$" "$build_dir/CMakeCache.txt"; then
  echo "CMake did not configure the requested CUDA setting." >&2
  exit 1
fi
build_jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
cmake --build "$build_dir" --parallel "$build_jobs"
install -m 755 "${build_dir}/src/foldseek" "$output_bin"

reported_version="$("$output_bin" version | tail -n 1)"
if [[ "$reported_version" != "$FOLDSEEK_COMMIT" ]]; then
  echo "Foldseek version mismatch: expected $FOLDSEEK_COMMIT, got $reported_version" >&2
  exit 1
fi
help_text="$("$output_bin" easy-multimersearch -h 2>&1 || true)"
for field in interfacelddt qchains tchains; do
  if ! grep -Eq "(^|[^[:alnum:]_])${field}([^[:alnum:]_]|$)" <<<"$help_text"; then
    echo "Foldseek does not expose required output field: $field" >&2
    exit 1
  fi
done
if [[ "$mode" == "gpu" ]] && ! grep -q -- "--gpu" <<<"$help_text"; then
  echo "Foldseek does not advertise GPU execution." >&2
  exit 1
fi

echo "foldseek_bin=${output_bin}"
echo "foldseek_commit=${FOLDSEEK_COMMIT}"
echo "device=${mode}"
if [[ "$mode" == "gpu" ]]; then
  echo "cuda_architectures=${cuda_architectures}"
  nvidia-smi -L
fi
