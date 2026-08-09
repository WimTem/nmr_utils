#!/usr/bin/env python3
"""
process_states_2d.py

Load, co-add, process, and plot a 2D SIMPSON HETCOR dataset acquired with
a States-type quadrature scheme in the indirect (t1) dimension.

Self-contained: the SIMPSON file I/O (previously in simpson_aux.py) is
merged in below, so this is the only file you need.

Expected row layout (produced by the corrected pulseq loop, inner
`foreach ph {x y}` at fixed t1, outer loop over t1):
    row 2k    -> x-phase excitation  (cosine-modulated,  "RE")
    row 2k+1  -> y-phase excitation  (sine-modulated,    "IM")
for k = 0 ... NI//2 - 1.

Usage:
    python process_states_2d.py <fid_dir> \
        [--gamma-ratio 0.25145] [--h-freq-mhz 300.0] \
        [--f1-offset-ppm 0.0] [--f2-offset-ppm 0.0] \
        [--lb1 300] [--lb2 300] [--si1 512] [--si2 512] \
        [--phase0-deg 180] [--plot-type contour]
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

# 13C/1H gyromagnetic ratio (gamma_13C / gamma_1H), used to derive the
# 13C Larmor frequency from the 1H spectrometer frequency instead of
# hardcoding it. Override --gamma-ratio for a different F2 nucleus.
GAMMA_RATIO_13C_1H = 10.7084 / 42.5774


# --------------------------------------------------------------------------
# SIMPSON file I/O (merged from simpson_aux.py)
# --------------------------------------------------------------------------

def load_simpson_2D_fid(file_name):
    """Read a SIMPSON ASCII .fid file and return (fid, t1, t2).

    fid is a complex array of shape (NI, NP); t1/t2 are time axes in seconds
    built from SW1/SW.
    """
    with open(file_name, "r") as file:
        first_line = file.readline().rstrip("\n")
        if first_line != "SIMP":
            print(f"Warning: '{file_name}' does not start with 'SIMP' header "
                  f"(got {first_line!r}) -- may not be a SIMPSON file")

        info = {}
        for line in file:
            line = line.rstrip("\n")
            if line == "DATA":
                break
            key, _, val = line.partition("=")
            info[key] = val

        if info.get("TYPE") != "FID":
            print(f"Warning: '{file_name}' TYPE is {info.get('TYPE')!r}, expected 'FID'")

        for required in ("NP", "SW", "SW1"):
            if required not in info:
                raise ValueError(f"'{file_name}': required field {required!r} not found in header")

        info["NP"] = int(info["NP"])
        info["SW"] = float(info["SW"])
        info["SW1"] = float(info["SW1"])
        info["NI"] = int(info["NI"]) if "NI" in info else 1

        n_points = info["NP"] * info["NI"]
        cplx = np.empty(n_points, dtype=complex)
        for i in range(n_points):
            line = file.readline().rstrip("\n")
            x_str, y_str = line.split()
            cplx[i] = complex(float(x_str), float(y_str))

    t1 = np.arange(info["NI"]) / info["SW1"]
    t2 = np.arange(info["NP"]) / info["SW"]
    fid = cplx.reshape((info["NI"], info["NP"]))
    return fid, t1, t2

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fid_dir", help="Directory containing .fid files to co-add")
    p.add_argument("--h-freq-mhz", type=float, default=300.0,
                   help="1H spectrometer (Larmor) frequency in MHz "
                        "(must match par(proton_frequency)/1e6 in the .spinsys)")
    p.add_argument("--gamma-ratio", type=float, default=GAMMA_RATIO_13C_1H,
                   help="gamma_X / gamma_1H ratio for the F2 nucleus")
    p.add_argument("--f1-offset-ppm", type=float, default=0.0,
                   help="Chemical-shift referencing offset applied to F1 (ppm)")
    p.add_argument("--f2-offset-ppm", type=float, default=0.0,
                   help="Chemical-shift referencing offset applied to F2 (ppm). "
                        "Use this to correct an unreferenced DFT shielding "
                        "scale to a calibrated chemical-shift scale.")
    p.add_argument("--lb1", type=float, default=300.0, help="Line broadening, F1 (Hz)")
    p.add_argument("--lb2", type=float, default=300.0, help="Line broadening, F2 (Hz)")
    p.add_argument("--si1", type=int, default=512, help="Final size, F1")
    p.add_argument("--si2", type=int, default=512, help="Final size, F2")
    p.add_argument("--sum", action="store_true",
                   help="Sum FIDs instead of averaging (default: average)")
    p.add_argument("--phase0-deg", type=float, default=180.0,
                   help="Zero-order phase correction applied to both States "
                        "channels, in degrees (default: 180 deg = pi)")
    p.add_argument("--plot-type", choices=["contour", "pcolormesh"], default="contour",
                   help="Plot style for the 2D spectrum (default: contour)")
    p.add_argument("--nlevels", type=int, default=24, help="Number of contour levels")
    p.add_argument("--lowest-level-frac", type=float, default=0.05,
                   help="Lowest contour level as a fraction of the spectrum max")
    p.add_argument("--level-scale", type=float, default=1.3,
                   help="Geometric growth factor between successive contour levels")
    return p.parse_args()


def sanitize_tag(path):
    """Turn a possibly-nested directory path into a safe filename fragment."""
    return os.path.basename(os.path.normpath(path))


# --------------------------------------------------------------------------
# Co-adding
# --------------------------------------------------------------------------

def load_and_combine(fid_dir, average=True):
    fid_paths = sorted(glob.glob(os.path.join(fid_dir, "*.fid")))
    if not fid_paths:
        raise FileNotFoundError(f"No .fid files found in '{fid_dir}'")

    print(f"Found {len(fid_paths)} .fid file(s) in folder {fid_dir}")

    combined_fid = None
    t1 = t2 = None
    ref_shape = None

    for path in fid_paths:
        fid, _t1, _t2 = load_simpson_2D_fid(path)

        if combined_fid is None:
            combined_fid = fid.astype(complex)
            t1, t2 = _t1, _t2
            ref_shape = fid.shape
        else:
            if fid.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch: {path} has shape {fid.shape}, "
                    f"expected {ref_shape}"
                )
            combined_fid += fid

    n_files = len(fid_paths)
    if average:
        combined_fid /= n_files
        print(f"Co-added {n_files} FID(s) (averaged).")
    else:
        print(f"Co-added {n_files} FID(s) (summed, absolute intensity preserved).")

    return combined_fid, t1, t2


def split_states_pairs(fid, t1):
    """
    Split rows into cosine- (RE) and sine- (IM) modulated sets, assuming
    row 2k and row 2k+1 were acquired at the SAME t1 value (see module
    docstring).
    """
    NI, NP = fid.shape
    if NI % 2 != 0:
        raise ValueError(f"NI={NI} must be even for States RE/IM pairs.")

    RE = fid[0::2, :].copy()   # cosine-modulated (x-phase)
    IM = fid[1::2, :].copy()   # sine-modulated   (y-phase)
    t1_pts = t1[0::2]          # RE and IM share the same t1 value per pair

    n_t1 = NI // 2
    t1_pts = np.arange(n_t1) / sw1

    return RE, IM, t1_pts


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------

def apodize(t_axis, lb_hz):
    """Cosine-squared + exponential apodization window for axis t_axis."""
    window = np.cos(t_axis / t_axis[-1] * np.pi / 2) ** 2
    window *= np.exp(-np.pi * lb_hz * t_axis)
    return window


def process(RE, IM, t1, t2, si1, si2, lb1, lb2, phase0_rad):
    # --- direct dimension (F2) ---
    apod2 = apodize(t2, lb2)
    RE = RE * apod2[np.newaxis, :]
    IM = IM * apod2[np.newaxis, :]

    # first-point correction for the DFT
    RE[:, 0] /= 2
    IM[:, 0] /= 2

    ftRE = np.fft.fftshift(np.fft.fft(RE, si2, axis=1), axes=1)
    ftIM = np.fft.fftshift(np.fft.fft(IM, si2, axis=1), axes=1)

    ftRE_phased = ftRE * np.exp(-1j * phase0_rad)
    ftIM_phased = ftIM * np.exp(-1j * phase0_rad)

    # States reconstruction: take the real part of each phase-shifted
    # spectrum and combine as the real/imaginary parts of a new complex
    # spectrum, which after the second FT yields pure absorption lineshapes
    ft1 = np.real(ftRE_phased) + 1j * np.real(ftIM_phased)

    # --- indirect dimension (F1) ---
    apod1 = apodize(t1, lb1)
    ft1 = ft1 * apod1[:, np.newaxis]
    ft1[0, :] /= 2

    ft2 = np.fft.fftshift(np.fft.fft(ft1, si1, axis=0), axes=0)
    return np.real(ft2)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    tag = sanitize_tag(args.fid_dir)

    fid, t1_full, t2 = load_and_combine(args.fid_dir, average=not args.sum)
    print(f"Raw FID shape: {fid.shape}")


    
    RE, IM, t1 = split_states_pairs(fid, t1_full)
    
    #SW1 = 1 / (t1_full[1] - t1_full[0])
    SW1 = 1 / (t1[1] - t1[0])
    SW2 = 1 / (t2[1] - t2[0])
    print(f"SW1 = {int(SW1)} Hz, SW2 = {int(SW2)} Hz")

    phase0_rad = np.deg2rad(args.phase0_deg)
    spec = process(RE, IM, t1, t2, args.si1, args.si2, args.lb1, args.lb2, phase0_rad)

    # --- frequency axes, ppm ---
    larmor_h = args.h_freq_mhz
    larmor_x = larmor_h * args.gamma_ratio

    f1 = np.linspace(-SW1 / 2, SW1 / 2, args.si1) / larmor_h + args.f1_offset_ppm
    f2 = np.linspace(-SW2 / 2, SW2 / 2, args.si2) / larmor_x + args.f2_offset_ppm
    F2, F1 = np.meshgrid(f2, f1)

    print(f"F1 span: {f1[-1] - f1[0]:.3f} ppm")
    print(f"F2 span: {f2[-1] - f2[0]:.3f} ppm")

    # --- normalize BEFORE computing contour levels, so levels match the
    # data actually being plotted ---
    #spec = spec / np.sum(np.abs(spec))
    spec_max = np.max(spec)
    spec_absmax = np.max(np.abs(spec))
    print(f"Max value of spec (normalized): {spec_max:.6g}")
    if spec_max <= 0:
        print("Warning: max of the processed spectrum is <= 0. This usually means "
              "--phase0-deg needs adjusting (try 0 instead of 180). Contour "
              "levels will be based on |spec| instead.")

    fig, ax = plt.subplots(figsize=(8, 4))

    if args.plot_type == "pcolormesh":
        mesh = ax.pcolormesh(F2, F1, spec, cmap="viridis", shading="auto")
        fig.colorbar(mesh, ax=ax, label="Normalized intensity")
    else:
        levs = (spec_absmax * args.lowest_level_frac) * args.level_scale ** np.arange(1, args.nlevels + 1)
        ax.contour(F2, F1, spec, levels=levs, colors="steelblue", linewidths=0.6)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # axis limits derived from the actual data range (with padding),
    # rather than hardcoded values that silently clip a different spin system
    pad2 = 0.05 * (f2[-1] - f2[0])
    pad1 = 0.05 * (f1[-1] - f1[0])
    #ax.set_xlim(f2[-1] + pad2, f2[0] - pad2)
    #ax.set_ylim(f1[-1] + pad1, f1[0] - pad1)
    ax.set_xlim(190,20)
    ax.set_ylim(20,-5)    

    ax.set_xlabel("F2 direct [ppm]")
    ax.set_ylabel("F1 indirect [ppm]")
    ax.set_title(f"States 2D spectrum — {tag}")
    fig.tight_layout()

    np.save(f"data_{tag}.npy", spec)
    np.save(f"x_{tag}.npy", F2)
    np.save(f"y_{tag}.npy", F1)

    out_png = f"states_2D_spec_{tag}.png"
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
