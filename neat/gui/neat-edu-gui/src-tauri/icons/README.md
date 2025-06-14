# Application Icons

This directory should contain the application icons in various formats:

- `32x32.png` - Small icon for taskbar
- `128x128.png` - Medium icon for desktop
- `128x128@2x.png` - High-DPI medium icon
- `icon.icns` - macOS icon format
- `icon.ico` - Windows icon format

You can generate these icons from a single source image using online tools or the Tauri icon generator.

For now, the application will use default Tauri icons. To customize:

1. Create a high-resolution source image (512x512 or larger)
2. Use an icon generator to create all required formats
3. Replace the placeholder files with your custom icons
4. Update the `tauri.conf.json` icon paths if needed

## Icon Design Guidelines

- Use a brain or neural network theme to match the NEAT educational platform
- Ensure good contrast and readability at small sizes
- Follow platform-specific design guidelines (iOS, Android, Windows, macOS)
- Test icons on different backgrounds and display densities