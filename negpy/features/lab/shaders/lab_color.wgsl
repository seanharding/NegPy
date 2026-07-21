// Chroma denoise, vibrance, saturation — mirrors CPU lab stage order before CLAHE.
// Decodes the encoded retouch buffer to scene-linear ProPhoto RGB.
struct LabUniforms {
    sharpen: f32,
    chroma_denoise: f32,
    saturation: f32,
    vibrance: f32,
    glow_amount: f32,
    halation_strength: f32,
    scale_factor: f32,
    _pad1: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LabUniforms;

const FIBONACCI_64 = array<vec2<f32>, 64>(
    vec2<f32>(0.088388, 0.000000),
    vec2<f32>(-0.112886, 0.103413),
    vec2<f32>(0.017279, -0.196886),
    vec2<f32>(0.142286, 0.185586),
    vec2<f32>(-0.261112, -0.046187),
    vec2<f32>(0.247348, -0.157342),
    vec2<f32>(-0.082733, 0.307763),
    vec2<f32>(-0.157781, -0.303797),
    vec2<f32>(0.342321, 0.125015),
    vec2<f32>(-0.356128, 0.147004),
    vec2<f32>(0.171677, -0.366864),
    vec2<f32>(0.126865, 0.404466),
    vec2<f32>(-0.382373, -0.221593),
    vec2<f32>(0.448567, -0.098616),
    vec2<f32>(-0.273753, 0.389386),
    vec2<f32>(-0.063243, -0.488045),
    vec2<f32>(0.388252, 0.327220),
    vec2<f32>(-0.522466, 0.021606),
    vec2<f32>(0.381099, -0.379244),
    vec2<f32>(-0.025497, 0.551396),
    vec2<f32>(-0.362617, -0.434536),
    vec2<f32>(0.574425, 0.077288),
    vec2<f32>(-0.486709, 0.338640),
    vec2<f32>(0.132997, -0.591185),
    vec2<f32>(0.307615, 0.536829),
    vec2<f32>(-0.601358, -0.191850),
    vec2<f32>(0.584143, -0.269889),
    vec2<f32>(-0.253065, 0.604686),
    vec2<f32>(-0.225855, -0.627935),
    vec2<f32>(0.600976, 0.315856),
    vec2<f32>(-0.667533, 0.175960),
    vec2<f32>(0.379431, -0.590102),
    vec2<f32>(0.120699, 0.702313),
    vec2<f32>(-0.572008, -0.442995),
    vec2<f32>(0.731702, -0.060620),
    vec2<f32>(-0.505760, 0.546712),
    vec2<f32>(0.003684, -0.755181),
    vec2<f32>(0.514305, 0.566946),
    vec2<f32>(-0.772295, -0.071576),
    vec2<f32>(0.625787, -0.474950),
    vec2<f32>(-0.142381, 0.782650),
    vec2<f32>(-0.428884, -0.681539),
    vec2<f32>(0.785920, 0.215388),
    vec2<f32>(-0.733486, 0.376413),
    vec2<f32>(0.289862, -0.781852),
    vec2<f32>(0.317911, 0.780942),
    vec2<f32>(-0.770264, -0.365042),
    vec2<f32>(0.823263, -0.253821),
    vec2<f32>(-0.440157, 0.751049),
    vec2<f32>(-0.184643, -0.859851),
    vec2<f32>(0.724177, 0.514422),
    vec2<f32>(-0.890157, 0.110939),
    vec2<f32>(0.587054, -0.689695),
    vec2<f32>(0.033320, 0.913689),
    vec2<f32>(-0.647727, -0.657276),
    vec2<f32>(0.930014, 0.047552),
    vec2<f32>(-0.724323, 0.598472),
    vec2<f32>(0.130975, -0.938767),
    vec2<f32>(0.542205, 0.787449),
    vec2<f32>(-0.939649, -0.216211),
    vec2<f32>(0.845937, -0.479274),
    vec2<f32>(-0.302492, 0.932436),
    vec2<f32>(-0.410097, -0.899101),
    vec2<f32>(0.916976, 0.389028)
);
const BLOOM_GAUSS_SUM = 27.668145;

fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return select(pow(e, vec3<f32>(1.8)), e / 16.0, e < vec3<f32>(0.03125));
}

fn load_lin(coords: vec2<i32>) -> vec3<f32> {
    return oetf_decode(textureLoad(input_tex, coords, 0).rgb);
}

fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);

    var x = r * 0.7976749 + g * 0.1351917 + b * 0.0313534;
    var y = r * 0.2880402 + g * 0.7118741 + b * 0.0000857;
    var z = r * 0.0000000 + g * 0.0000000 + b * 0.8252100;

    x = x / 0.96422;
    y = y / 1.00000;
    z = z / 0.82521;

    if (x > 0.008856) { x = pow(x, 1.0/3.0); } else { x = (7.787 * x) + (16.0 / 116.0); }
    if (y > 0.008856) { y = pow(y, 1.0/3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    if (z > 0.008856) { z = pow(z, 1.0/3.0); } else { z = (7.787 * z) + (16.0 / 116.0); }

    let l = (116.0 * y) - 16.0;
    let a = 500.0 * (x - y);
    let b_lab = 200.0 * (y - z);

    return vec3<f32>(l, a, b_lab);
}

fn lab_to_rgb(lab: vec3<f32>) -> vec3<f32> {
    var y = (lab.x + 16.0) / 116.0;
    var x = lab.y / 500.0 + y;
    var z = y - lab.z / 200.0;

    if (pow(x, 3.0) > 0.008856) { x = pow(x, 3.0); } else { x = (x - 16.0 / 116.0) / 7.787; }
    if (pow(y, 3.0) > 0.008856) { y = pow(y, 3.0); } else { y = (y - 16.0 / 116.0) / 7.787; }
    if (pow(z, 3.0) > 0.008856) { z = pow(z, 3.0); } else { z = (z - 16.0 / 116.0) / 7.787; }

    x = x * 0.96422;
    y = y * 1.00000;
    z = z * 0.82521;

    let r = x * 1.3459433 + y * -0.2556075 + z * -0.0511118;
    let g = x * -0.5445989 + y * 1.5081673 + z * 0.0205351;
    let b = x * 0.0000000 + y * 0.0000000 + z * 1.2118128;

    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = load_lin(coords);

    if (params.chroma_denoise > 0.0) {
        let radius = 2.0 * params.chroma_denoise * params.scale_factor;
        if (radius >= 0.5) {
            let current_lab = rgb_to_lab(color);
            var blur_ab = vec2<f32>(0.0);
            for (var tap = 0; tap < 64; tap++) {
                let offset = FIBONACCI_64[tap];
                let s_coord = clamp(coords + vec2<i32>(offset * radius), vec2<i32>(0), vec2<i32>(dims) - 1);
                let s_lab = rgb_to_lab(load_lin(s_coord));
                let r = length(offset);
                let w = exp(-r * r * 2.0);
                blur_ab += s_lab.yz * w;
            }
            blur_ab = blur_ab / BLOOM_GAUSS_SUM;
            color = lab_to_rgb(vec3<f32>(current_lab.x, blur_ab.x, blur_ab.y));
        }
    }

    if (params.vibrance != 1.0) {
        var lab = rgb_to_lab(color);
        let chroma = length(lab.yz);
        let muted_mask = clamp(1.0 - (chroma / 60.0), 0.0, 1.0);
        let boost = (params.vibrance - 1.0) * muted_mask;
        lab.y = lab.y * (1.0 + boost);
        lab.z = lab.z * (1.0 + boost);
        color = lab_to_rgb(lab);
    }

    if (params.saturation != 1.0) {
        var lab = rgb_to_lab(color);
        lab.y = lab.y * params.saturation;
        lab.z = lab.z * params.saturation;
        color = lab_to_rgb(lab);
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
