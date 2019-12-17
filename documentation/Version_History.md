# Φ<sub>*Flow*</sub> Version History

## Version 1.0.x

### Version 1.0.1

Version Highlights:
- **GUI-2.0**: Improved web interface with multiple pages and experimental 3D WebGL renderer
- **Documentation**: Google Colab notebooks and better demo descriptions

Breaking Changes: None

Deprecations:
- `App.stride`: Use framerate instead of stride in App constructor
- `world.add`: Explicitly create a Physics object when adding a state to the world.

### Version 1.0.0

- Public release on GitHub
- Open-source with MIT license
- Use Travis-CI for testing


## Pre-release version

Major refactoring was performed before release, renaming (among others): Smoke &rarr; Fluid, anytype &rarr; unsafe.

Version 0.4 introduced the Field API, bringing with it a new data layout for staggered grids.

Version 0.3 introduced structs and used them to split state and physics into different objects.
