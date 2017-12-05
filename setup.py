from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools
import subprocess

__version__ = '0.0.1'

# Set up a few file paths
abspath = os.path.abspath(os.path.dirname("__file__"))
build_dir = os.path.join(abspath, "build")

tblis_build_dir = os.path.join(build_dir, "tblis_build")
tblis_install_dir = os.path.join(build_dir, "tblis_install")
tblis_config_file = os.path.join(abspath, "configure")


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'tblis',
        ['src/pytblis/main.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(tblis_install_dir, "include"),
            os.path.join(tblis_install_dir, "include", "tblis"),

        ],
        library_dirs=[os.path.join(tblis_install_dir, "lib")],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

def autoconf():
    """
    A simple program to run the autotools configure and make in a python-friendly directory
    """

    for folder in [build_dir, tblis_build_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    os.chdir(tblis_build_dir)

    print("\nRunning configure...")
    retcode = subprocess.Popen([tblis_config_file, "--prefix=" + tblis_install_dir], bufsize=0,
                            stdout=subprocess.PIPE, universal_newlines=True)

    while True:
        data = retcode.stdout.readline()
        if not data:
            break

        if "config.status" in data:
            print(data.strip())

    print("\nBuilding...")
    retcode = subprocess.Popen(['make', '-j2'], bufsize=0, stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE, universal_newlines=True)
    ctestout = ''
    while True:
        data = retcode.stdout.readline()
        if not data:
            break

        if "CC" in data or "CXX" in data:
            print(data.strip())

    print("\nInstalling...")
    retcode = subprocess.Popen(['make', 'install'], bufsize=0, stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE, universal_newlines=True)
    ctestout = ''
    while True:
        data = retcode.stdout.readline()
        if not data:
            break
        if ("libtool: install:" in data) or ("Making "):
            print(data.strip())

    os.chdir(abspath)



class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += ['-Wno-reorder', '-Wno-unused-variable']

    def build_extensions(self):

        # First build TBLIS itself
#        autoconf()

        print("\nBuild Python wrapper\n")
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='TBLIS',
    version=__version__,
    author='Devin Matthews and Daniel G. A. Smith',
    author_email='dgasmith@vt.edu',
    url='https://github.com/devinamatthews/tblis',
    description='A Python interface to TBLIS',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['numpy', 'pybind11>=2.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
