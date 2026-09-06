project('vector', 'c',
  version: '0.8.6',
  license: 'PostgreSQL',
  default_options: [
    'c_std=c99',
    'warning_level=2',
    'buildtype=release',
  ]
)

cc = meson.get_compiler('c')
host_system = host_machine.system()

# Discover PostgreSQL paths and flags via pg_config
pg_config = find_program('pg_config', required: true)
pg_version = run_command(pg_config, '--version', check: true).stdout().strip()
pg_includedir_server = run_command(pg_config, '--includedir-server', check: true).stdout().strip()
pg_includedir = run_command(pg_config, '--includedir', check: true).stdout().strip()
pg_pkglibdir = run_command(pg_config, '--pkglibdir', check: true).stdout().strip()
pg_sharedir = run_command(pg_config, '--sharedir', check: true).stdout().strip()
pg_bindir = run_command(pg_config, '--bindir', check: true).stdout().strip()

pg_cppflags = run_command(pg_config, '--cppflags', check: false).stdout().strip().split()
pg_cflags_sl = run_command(pg_config, '--cflags_sl', check: false).stdout().strip().split()
pg_ldflags = run_command(pg_config, '--ldflags', check: false).stdout().strip().split()
pg_ldflags_sl = run_command(pg_config, '--ldflags_sl', check: false).stdout().strip().split()

# Include directories
inc_dirs = [
  include_directories('src'),
  include_directories(pg_includedir_server),
  include_directories(pg_includedir),
]

# Check if internal include dir exists
fs = import('fs')
if fs.is_dir(pg_includedir / 'internal')
  inc_dirs += include_directories(pg_includedir / 'internal')
endif

# Vector optimization flags matching Makefile
c_args = [
  '-ftree-vectorize',
  '-fassociative-math',
  '-fno-signed-zeros',
  '-fno-trapping-math',
  '-ffp-contract=fast',
] + pg_cppflags + pg_cflags_sl

# Target architecture optimizations
if host_system != 'darwin' and host_machine.cpu_family() not in ['ppc64', 'riscv64']
  if cc.has_argument('-march=native')
    c_args += '-march=native'
  endif
endif

link_args = pg_ldflags + pg_ldflags_sl
if host_system == 'darwin'
  link_args += ['-undefined', 'dynamic_lookup']
endif

vector_sources = files(
  'src/bitutils.c',
  'src/bitvec.c',
  'src/clustering.c',
  'src/halfutils.c',
  'src/halfvec.c',
  'src/hnsw.c',
  'src/hnswbuild.c',
  'src/hnswinsert.c',
  'src/hnswscan.c',
  'src/hnswutils.c',
  'src/hnswvacuum.c',
  'src/ivfbuild.c',
  'src/ivfflat.c',
  'src/ivfinsert.c',
  'src/ivfkmeans.c',
  'src/ivfscan.c',
  'src/ivfutils.c',
  'src/ivfvacuum.c',
  'src/sparsevec.c',
  'src/vector.c',
)

vector_module = shared_module(
  'vector',
  vector_sources,
  include_directories: inc_dirs,
  c_args: c_args,
  link_args: link_args,
  install: true,
  install_dir: pg_pkglibdir,
  name_prefix: '',
)

# Extension control and SQL files
install_data(
  'vector.control',
  install_dir: pg_sharedir / 'extension',
)

# Generate versioned extension SQL file
sql_file = custom_target(
  'vector--' + meson.project_version() + '.sql',
  input: 'sql/vector.sql',
  output: 'vector--' + meson.project_version() + '.sql',
  command: [find_program('cp', 'copy'), '@INPUT@', '@OUTPUT@'],
  install: true,
  install_dir: pg_sharedir / 'extension',
)