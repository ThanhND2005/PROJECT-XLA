// GLCM Texture 0.4 wrapper; input: original 8-bit PNG images.
// Default configuration: 16 gray levels, distance 1, angle 0 degrees.
// Optional argument: input_directory|output_csv
requires("1.54p");
levels = 16; distance = 1; angle = 0;
processed = 0; skipped = 0;
arg = getArgument();
if (arg == "") {
    root = getDirectory("Original 8-bit PNG images");
    outDir = getDirectory("CSV output directory");
    outCsv = outDir + "glcm_L16_d1_a0_" + d2s(getTime(), 0) + ".csv";
} else {
    parts = split(arg, "|");
    if (parts.length != 2) exit("Expected input_directory|output_csv");
    root = parts[0]; outCsv = parts[1];
}
if (!endsWith(root, File.separator)) root += File.separator;
if (!File.isDirectory(root)) exit("Input directory does not exist");
if (File.exists(outCsv)) exit("Output CSV already exists");
header = "image,width,height,levels,distance,angle,";
header += "asm,contrast,correlation,energy,homogeneity,entropy,sum,status\n";
File.saveString(header, outCsv);
setBatchMode(true);
scanFolder(root, "", outCsv, levels, distance, angle);
setBatchMode(false);
print("Processed=" + processed + "; skipped=" + skipped);
print("CSV=" + outCsv);

function scanFolder(folder, relative, csv, levels, distance, angle) {
    names = getFileList(folder);
    Array.sort(names);
    for (k = 0; k < names.length; k++) {
        name = names[k]; path = folder + name;
        if (File.isDirectory(path)) {
            scanFolder(path, relative + name, csv, levels, distance, angle);
        } else if (endsWith(toLowerCase(name), ".png")) {
            extractOne(path, relative + name, csv, levels, distance, angle);
        }
    }
}

function extractOne(path, label, csv, levels, distance, angle) {
    open(path);
    imageId = getImageID();
    getDimensions(w, h, channels, slices, frames);
    if (bitDepth != 8 || channels * slices * frames != 1) {
        print("Skipped unsupported image: " + label);
        skipped++;
        setOption("Changes", false); close();
        return;
    }
    run("Select None");
    divisor = 256 / levels;
    run("Macro...", "code=v=floor(v/" + divisor + ")");
    options = "enter=" + distance + " select=[" + angle + " degrees]";
    options += " angular contrast correlation inverse entropy";
    run("GLCM Texture", options);
    if (nResults != 1) exit("Unexpected result row count: " + label);
    asm = getResult("Angular Second Moment", 0);
    contrast = getResult("Contrast", 0);
    correlation = getResult("Correlation", 0);
    homogeneity = getResult("Inverse Difference Moment   ", 0);
    entropy = getResult("Entropy", 0);
    total = getResult("Sum of all GLCM elements", 0);
    if (isNaN(asm) || isNaN(contrast) || isNaN(homogeneity))
        exit("Missing plugin output: " + label);
    status = "ok";
    if (isNaN(correlation)) status = "undefined_correlation";
    line = "" + csvQuote(label) + "," + w + "," + h + "," + levels;
    line += "," + distance + "," + angle + "," + d2s(asm, 6);
    line += "," + d2s(contrast, 6) + "," + d2s(correlation, 6);
    line += "," + d2s(sqrt(asm), 6) + "," + d2s(homogeneity, 6);
    line += "," + d2s(entropy, 6) + "," + d2s(total, 6);
    File.append(line + "," + status, csv);
    processed++;
    selectImage(imageId);
    setOption("Changes", false); close();
}

function csvQuote(value) {
    return "\"" + replace(value, "\"", "\"\"") + "\"";
}
