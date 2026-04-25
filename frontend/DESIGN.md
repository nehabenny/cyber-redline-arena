---
name: Cyber-Vault Design System
colors:
  surface: '#10131a'
  surface-dim: '#10131a'
  surface-bright: '#363940'
  surface-container-lowest: '#0b0e14'
  surface-container-low: '#191c22'
  surface-container: '#1d2026'
  surface-container-high: '#272a31'
  surface-container-highest: '#32353c'
  on-surface: '#e1e2eb'
  on-surface-variant: '#c4c9ac'
  inverse-surface: '#e1e2eb'
  inverse-on-surface: '#2e3037'
  outline: '#8e9379'
  outline-variant: '#444933'
  surface-tint: '#abd600'
  primary: '#ffffff'
  on-primary: '#283500'
  primary-container: '#c3f400'
  on-primary-container: '#556d00'
  inverse-primary: '#506600'
  secondary: '#bdf4ff'
  on-secondary: '#00363d'
  secondary-container: '#00e3fd'
  on-secondary-container: '#00616d'
  tertiary: '#ffffff'
  on-tertiary: '#621100'
  tertiary-container: '#ffdad2'
  on-tertiary-container: '#bf2b00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#c3f400'
  primary-fixed-dim: '#abd600'
  on-primary-fixed: '#161e00'
  on-primary-fixed-variant: '#3c4d00'
  secondary-fixed: '#9cf0ff'
  secondary-fixed-dim: '#00daf3'
  on-secondary-fixed: '#001f24'
  on-secondary-fixed-variant: '#004f58'
  tertiary-fixed: '#ffdad2'
  tertiary-fixed-dim: '#ffb4a2'
  on-tertiary-fixed: '#3c0700'
  on-tertiary-fixed-variant: '#8a1d00'
  background: '#10131a'
  on-background: '#e1e2eb'
  surface-variant: '#32353c'
typography:
  headline-xl:
    fontFamily: Space Grotesk
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: 0.02em
  data-mono:
    fontFamily: monospace
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.5'
    letterSpacing: 0.05em
  body-main:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: 0em
  label-caps:
    fontFamily: Space Grotesk
    fontSize: 12px
    fontWeight: '700'
    lineHeight: '1'
    letterSpacing: 0.1em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin: 24px
  panel-padding: 20px
  container-max: 1920px
---

## Brand & Style

This design system is engineered for high-stakes Security Operations Centers, evoking the atmosphere of a digital fortress. The brand personality is authoritative, vigilant, and high-velocity. It targets elite cybersecurity analysts who require deep focus and rapid data synthesis.

The aesthetic follows a **Tactical Glassmorphism** movement. It combines the structural rigidity of a physical vault with the ethereal quality of advanced software. Expect deep shadows, layered transparency, and high-frequency data density. The emotional response is one of absolute control and impenetrable security, achieved through a "HUD" (Heads-Up Display) visual language that prioritizes signal over noise.

## Colors

The palette is anchored in a deep, "Abyssal Black" (#0B0E14) to minimize eye strain during long shifts. 

- **Cyber Lime (#CCFF00):** Used for primary actions, system-ready states, and critical "safe" pathing. 
- **Electric Blue (#00E5FF):** Utilized for data visualization, active scanning indicators, and informational UI elements.
- **Alert Red (#FF3D00):** Reserved exclusively for breaches and critical system failures.
- **Glass Accents:** Transparent layers use white or primary-tinted strokes at very low opacity (10-15%) to define edges without adding visual bulk.

## Typography

This design system utilizes a dual-font strategy to balance legibility with a technical aesthetic. 

- **UI & Content:** `Inter` is the primary workhorse, ensuring that complex documentation and long-form logs remain readable.
- **Technical & Headlines:** `Space Grotesk` provides a futuristic, geometric feel for titles and labels. 
- **Data Tables:** For IP addresses, hash values, and timestamps, use a standard system Monospace (fallback to JetBrains Mono where available) to ensure perfect vertical alignment and character distinction. 

All labels should be treated with high tracking (letter-spacing) to mimic military-grade hardware interfaces.

## Layout & Spacing

The design system employs a **Modular Tactical Grid**. This is a 12-column fluid system designed for 4K and Ultra-Wide displays common in SOC environments.

Spacing is based on a strict 4px baseline grid to maintain mathematical precision. Components should be docked into "Zones" (e.g., Threat Map Zone, Log Stream Zone) with 16px gutters between panels. 

Layouts should maximize "Information Density"—minimize empty whitespace in favor of structural lines and data-rich modules. Panels can be collapsible to allow analysts to focus on specific threat vectors.

## Elevation & Depth

Depth is not communicated through traditional shadows, but through **Luminous Layers** and **Backdrop Blurs**.

1.  **Base Layer:** The darkest surface (#0B0E14), representing the "void" or background.
2.  **Panel Layer:** Semi-transparent (60-80% opacity) with a `backdrop-filter: blur(20px)`. This creates the glass effect.
3.  **Active Stroke:** Elements in focus or panels containing active threats feature a 1px solid border of Cyber Lime or Electric Blue with a subtle `box-shadow` glow (0px 0px 8px) in the same color.
4.  **Overlay:** Modals and dropdowns use a higher opacity and a brighter border to appear physically closer to the user.

## Shapes

The shape language is **Precision-Engineered**. We use a "Soft-Sharp" approach (Level 1 roundedness).

Standard components feature a 4px corner radius. This is enough to feel modern and premium, but sharp enough to maintain a serious, military-grade profile. For specific "Status" chips or "Threat Level" indicators, use a 0px radius (Sharp) on one side and 4px on the other to create a directional, "arrow" or "tab" aesthetic.

## Components

### Buttons
Buttons are secondary-stroke only by default, filling with a solid neon glow on hover. The "Primary" button uses a Cyber Lime background with black text for maximum contrast. All buttons should have a `text-transform: uppercase` and `letter-spacing: 0.05em`.

### Tactical Panels (Cards)
Cards are the primary container. They must feature a 1px "Glass Stroke" (rgba 255,255,255, 0.1) and a header section with a distinct background tint. Add a small "corner bracket" SVG in the four corners to enhance the "Cyber-Vault" feel.

### Status Chips
Chips for "High," "Medium," and "Low" threats should use high-saturation backgrounds (Red, Orange, Blue) with a blurred "inner glow" to appear like physical LED indicators.

### Data Inputs
Input fields are minimal, consisting of a bottom border that illuminates in Electric Blue when focused. The cursor should be a solid block rather than a line to maintain the monospace/technical aesthetic.

### Additional Components
- **Scan-line Overlays:** A subtle, animated horizontal line moving across data-heavy panels.
- **Health Gauges:** Circular, segmented progress bars for CPU, Memory, and Network load.
- **Breadcrumbs:** Connected by chevrons ( `>` ) to show the file path or network drill-down.