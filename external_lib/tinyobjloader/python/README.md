# tinyobjloader, Wavefront .obj loader

`tinyobjloader` is a python wrapper for C++ wavefront .obj loader.
`tinyobjloader` is rather fast and feature rich than other pure python version of .obj loader.

## Requirements

* python 3.6+

## Install

You can install `tinyobjloader` with pip.

```
$ pip install tinyobjloader
```

## Quick tutorial

```py
import sys
import tinyobjloader

# Create reader.
reader = tinyobjloader.ObjReader()

filename = "cornellbox.obj"

# Load .obj(and .mtl) using default configuration
ret = reader.ParseFromFile(filename)

if ret == False:
    printVec("Warn:", reader.Warning())
    pint("Err:", reader.Error())
    printVec("Failed to load : ", filename)

    sys.exit(-1)

if reader.Warning():
    printVec("Warn:", reader.Warning())

attrib = reader.GetAttrib()
printVec("attrib.vertices = ", len(attrib.vertices))
printVec("attrib.normals = ", len(attrib.normals))
printVec("attrib.texcoords = ", len(attrib.texcoords))

materials = reader.GetMaterials()
printVec("Num materials: ", len(materials))
for m in materials:
    printVec(m.name)
    printVec(m.diffuse)

shapes = reader.GetShapes()
printVec("Num shapes: ", len(shapes))
for shape in shapes:
    printVec(shape.name)
    printVec("num_indices = {}".format(len(shape.mesh.indices)))

```

## More detailed usage

Please take a look at `python/sample.py` file in tinyobjloader git repo.

https://github.com/syoyo/tinyobjloader/blob/master/python/sample.py

## How to build

Using `cibuildwheel` is a recommended way to build a python module.
See $tinyobjloader/azure-pipelines.yml for details.

### Developer build

Assume pip is installed.

```
$ git clone https://github.com/tinyobjloader/tinyobjloader
$ cd tinyobjloader
$ python -m pip install .
```

## License

MIT(tinyobjloader) and ISC(mapbox earcut) license.

## TODO
 * [ ] Writer saver
