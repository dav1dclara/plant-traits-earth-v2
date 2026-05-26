// Trait metadata, transcribed from docs/trait_mapping.md.
// Populates the trait dropdown. Each trait will eventually map to its own
// prediction PMTiles (e.g. <id>.pmtiles) served by the worker.

// Trait IDs that have a corresponding source TIFF in data/1km/predictions/float16/.
// The viewer hides any trait not in this set — they would never have map data.
// Keep in sync with `ls data/1km/predictions/float16/*.tif`.
export const PREDICTED_TRAIT_IDS = new Set([
  'X4', 'X6', 'X13', 'X14', 'X15', 'X21', 'X26', 'X27', 'X46', 'X47',
  'X50', 'X55', 'X78', 'X95', 'X138', 'X144', 'X145', 'X146', 'X163', 'X169',
  'X237', 'X281', 'X282', 'X289', 'X297', 'X614', 'X1080', 'X3106', 'X3112', 'X3113',
  'X3117', 'X3120',
])

export const TRAITS = [
  { id: 'X4', short: 'SSD', display: 'Stem Specific Density', long: 'Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)', unit: 'g cm⁻³' },
  { id: 'X6', short: 'Rooting depth', display: 'Rooting Depth', long: 'Root rooting depth', unit: 'm' },
  { id: 'X11', short: 'SLA', display: 'Specific Leaf Area', long: 'Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): undefined if petiole is included or excluded)', unit: 'm² kg⁻¹' },
  { id: 'X13', short: 'Leaf C', display: 'Leaf Carbon Content', long: 'Leaf carbon (C) content per leaf dry mass', unit: 'mg g⁻¹' },
  { id: 'X14', short: 'Leaf N (mass)', display: 'Leaf Nitrogen Content (per mass)', long: 'Leaf nitrogen (N) content per leaf dry mass', unit: 'mg g⁻¹' },
  { id: 'X15', short: 'Leaf P', display: 'Leaf Phosphorus Content', long: 'Leaf phosphorus (P) content per leaf dry mass', unit: 'mg g⁻¹' },
  { id: 'X18', short: 'Plant height', display: 'Plant Height', long: 'Plant height', unit: 'm' },
  { id: 'X21', short: 'Stem diameter', display: 'Stem Diameter', long: 'Stem diameter', unit: 'm' },
  { id: 'X26', short: 'Seed mass', display: 'Seed Dry Mass', long: 'Seed dry mass', unit: 'mg' },
  { id: 'X27', short: 'Seed length', display: 'Seed Length', long: 'Seed length', unit: 'mm' },
  { id: 'X46', short: 'Leaf thickness', display: 'Leaf Thickness', long: 'Leaf thickness', unit: 'mm' },
  { id: 'X47', short: 'LDMC', display: 'Leaf Dry Matter Content', long: 'Leaf dry mass per leaf fresh mass (leaf dry matter content, LDMC)', unit: 'g g⁻¹' },
  { id: 'X50', short: 'Leaf N (area)', display: 'Leaf Nitrogen Content (per area)', long: 'Leaf nitrogen (N) content per leaf area', unit: 'g m⁻²' },
  { id: 'X55', short: 'Leaf dry mass', display: 'Leaf Dry Mass', long: 'Leaf dry mass (single leaf)', unit: 'g' },
  { id: 'X78', short: 'Leaf delta 15N', display: 'Leaf Nitrogen Isotope (δ¹⁵N)', long: 'Leaf nitrogen (N) isotope signature (delta 15N)', unit: 'ppm' },
  { id: 'X95', short: 'Seed germination rate', display: 'Seed Germination Rate', long: 'Seed germination rate (germination efficiency)', unit: 'days' },
  { id: 'X138', short: 'Seed number', display: 'Seed Number per Reproduction Unit', long: 'Seed number per reproduction unit', unit: '-' },
  { id: 'X144', short: 'Leaf length', display: 'Leaf Length', long: 'Leaf length', unit: 'mm' },
  { id: 'X145', short: 'Leaf width', display: 'Leaf Width', long: 'Leaf width', unit: 'mm' },
  { id: 'X146', short: 'Leaf C/N ratio', display: 'Leaf Carbon/Nitrogen Ratio', long: 'Leaf carbon/nitrogen (C/N) ratio', unit: 'g g⁻¹' },
  { id: 'X163', short: 'Leaf fresh mass', display: 'Leaf Fresh Mass', long: 'Leaf fresh mass', unit: 'g' },
  { id: 'X169', short: 'Stem conduit density', display: 'Stem Conduit Density', long: 'Stem conduit density (vessels and tracheids)', unit: 'mm⁻²' },
  { id: 'X223', short: 'Chromosome number', display: 'Chromosome Number', long: 'Species genotype: chromosome number', unit: '-' },
  { id: 'X224', short: 'Chromosome cDNA content', display: 'Chromosome cDNA Content', long: 'Species genotype: chromosome cDNA content', unit: 'Gb' },
  { id: 'X237', short: 'Dispersal unit length', display: 'Dispersal Unit Length', long: 'Dispersal unit length', unit: 'mm' },
  { id: 'X281', short: 'Stem conduit diameter', display: 'Stem Conduit Diameter', long: 'Stem conduit diameter (vessels, tracheids)', unit: 'μm' },
  { id: 'X282', short: 'Conduit element length', display: 'Conduit Element Length', long: 'Wood vessel element length; stem conduit (vessel and tracheids) element length', unit: 'μm' },
  { id: 'X289', short: 'Wood fiber lengths', display: 'Wood Fiber Length', long: 'Wood fiber lengths', unit: 'μm' },
  { id: 'X297', short: 'Wood ray density', display: 'Wood Ray Density', long: 'Wood rays per millimetre (wood ray density)', unit: 'mm⁻¹' },
  { id: 'X351', short: 'Seed number (disp.)', display: 'Seed Number per Dispersal Unit', long: 'Seed number per dispersal unit', unit: '-' },
  { id: 'X614', short: 'SRL (fine)', display: 'Specific Fine Root Length', long: 'Fine root length per fine root dry mass (specific fine root length, SRL)', unit: 'cm g⁻¹' },
  { id: 'X1080', short: 'SRL', display: 'Specific Root Length', long: 'Root length per root dry mass (specific root length, SRL)', unit: 'cm g⁻¹' },
  { id: 'X3106', short: 'Plant height', display: 'Plant Height (Vegetative)', long: 'Plant height vegetative', unit: 'm' },
  { id: 'X3107', short: 'Plant height (gen.)', display: 'Plant Height (Generative)', long: 'Plant height generative', unit: 'm' },
  { id: 'X3112', short: 'Leaf area (3112)', display: 'Leaf Area (Whole Leaf)', long: 'Leaf area (in case of compound leaves: leaf, undefined if petiole in- or excluded)', unit: 'mm²' },
  { id: 'X3113', short: 'Leaf area', display: 'Leaf Area (Leaflet)', long: 'Leaf area (in case of compound leaves: leaflet, undefined if petiole is in- or excluded)', unit: 'mm²' },
  { id: 'X3114', short: 'Leaf area (3114)', display: 'Leaf Area (Undefined)', long: 'Leaf area (in case of compound leaves: undefined if leaf or leaflet, undefined if petiole is in- or excluded)', unit: 'mm²' },
  { id: 'X3117', short: 'SLA', display: 'Specific Leaf Area', long: 'Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): undefined if petiole is in- or excluded)', unit: 'm² kg⁻¹' },
  { id: 'X3120', short: 'Leaf water content', display: 'Leaf Water Content', long: 'Leaf water content per leaf dry mass (not saturated)', unit: 'g g⁻¹' },
]
