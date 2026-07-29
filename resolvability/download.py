"""Automatic download of the released data and checkpoints.

Everything the paper's figures read is hosted publicly and fetched on first use, so a fresh
clone reproduces every figure without regenerating anything. Each script calls :func:`ensure`
on the files it needs; the call is a no-op once the file is on disk.

Two tiers, by what they let you do:

``data``         the seismic and groundwater inputs the pipelines and renderers open -- the
                 illumination spectra and probe basis, the slim evaluation window, the small
                 groundwater dataset, and the result caches (``results/``) the figures read --
                 together with the full seismic training and evaluation archives, which are
                 large and needed only to retrain a prior or recompute an evaluation window.
``checkpoints``  the trained priors: three seeds per arm for the seismic and groundwater
                 examples. Needed to re-run sampling.

The URLs below are public direct-download links; see the dataset section of the README.
"""

from __future__ import annotations

import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# repo-relative path -> (tier, direct-download URL).
REGISTRY: dict[str, tuple[str, str]] = {
    # --- results: the resolvability statement, certify, augment, and cutoff caches ---
    "results/darcy_certify.npz":
        ("data", "https://www.dropbox.com/scl/fi/ukjp38cxdpabvbeyj2hdh/darcy_certify.npz?rlkey=bv20n0nal9akcjtefrzz2rquq&dl=1"),
    "results/darcy_core_channel.npz":
        ("data", "https://www.dropbox.com/scl/fi/0bvsvy5pgrysabiwjo617/darcy_core_channel.npz?rlkey=n6rvyia4z48k6hgzwai358155&dl=1"),
    "results/darcy_joint_archive.npz":
        ("data", "https://www.dropbox.com/scl/fi/15ijik3kp3vg2u4m5bxdy/darcy_joint_archive.npz?rlkey=ikve3rkwumur52ke2zfgisgz0&dl=1"),
    "results/darcy_augment_controls.npz":
        ("data", "https://www.dropbox.com/scl/fi/4isnsyj0r5dah6td5hc2k/darcy_augment_controls.npz?rlkey=7sk9cnm6pypvddqlwq4zc9qxg&dl=1"),
    "results/seismic_cutoff_sweep.npz":
        ("data", "https://www.dropbox.com/scl/fi/apdycvbocloyznfor3dcp/seismic_cutoff_sweep.npz?rlkey=k4j5hgk8ciwjbfvbahne6ui78&dl=1"),
    "results/seismic_gallery_samples.npz":
        ("data", "https://www.dropbox.com/scl/fi/kz57x9smh85074pvjvmbc/seismic_gallery_samples.npz?rlkey=71vljqlqzvkgbpjv9hf908889&dl=1"),
    # --- checkpoints: the groundwater flows, three seeds per arm plus the jointly curated arm ---
    "data/checkpoints/darcy_flows/oracle_s10.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/op4aifuxtvy1epq7bsp78/oracle_s10.pth?rlkey=iw8vxxcu42xdj1u5pod0apdrr&dl=1"),
    "data/checkpoints/darcy_flows/oracle_s11.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/qpokougq7451l9casv0m8/oracle_s11.pth?rlkey=4kriulnnl4v5drniv2twjmnp9&dl=1"),
    "data/checkpoints/darcy_flows/oracle_s12.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/7q8ajyyuyj72pwuysza5h/oracle_s12.pth?rlkey=i4qh7alhnavqbistdww7ocqrh&dl=1"),
    "data/checkpoints/darcy_flows/curated_s10.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/1s9q6gvd5lgnw85dl9xhd/curated_s10.pth?rlkey=ehat101cqa6l8a6487bmt7jc6&dl=1"),
    "data/checkpoints/darcy_flows/curated_s11.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/5k1vnbvrh1mxkyb94gtp7/curated_s11.pth?rlkey=gt0uwe36pidk7pi66yvnwjwlw&dl=1"),
    "data/checkpoints/darcy_flows/curated_s12.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/zad61imk1hkt85ejvzrcl/curated_s12.pth?rlkey=ln6a6evgb54f5upk90yoj5usg&dl=1"),
    "data/checkpoints/darcy_flows/joint_s10.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/aim6r2sdk5uj4mnk87066/joint_s10.pth?rlkey=81eq49o0w61838bp4lys6tbfp&dl=1"),
    "data/checkpoints/darcy_flows/joint_s11.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/xubig3cnhyvn5e7pfl13e/joint_s11.pth?rlkey=ipgvcds5e66cynbanvniryn9w&dl=1"),
    "data/checkpoints/darcy_flows/joint_s12.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/3ky2io9aprffe9w6udfjs/joint_s12.pth?rlkey=btfvhc035mk3gjt9xwz919t7d&dl=1"),
    # --- data: seismic and groundwater inputs ---
    "data/darcy/darcy_laundering.h5":
        ("data", "https://www.dropbox.com/scl/fi/2zdbajsv9hxlkmlwh33sr/darcy_laundering.h5?rlkey=koet9vijaijmza4uxznwguucy&dl=1"),
    "data/seismic/effect_existence_spectrum.npz":
        ("data", "https://www.dropbox.com/scl/fi/jcrlbeee9shbpmlmcso1b/effect_existence_spectrum.npz?rlkey=bjaskwfe93u5dnipoibrxwm2w&dl=1"),
    "data/seismic/eval_window.npz":
        ("data", "https://www.dropbox.com/scl/fi/zizj90uk39q24g03pfsi2/eval_window.npz?rlkey=l1y9yvsrnelpflxrne035wqul&dl=1"),
    "data/seismic/psf_illum_N256.npz":
        ("data", "https://www.dropbox.com/scl/fi/wmtu06k73eif53ki2sd9k/psf_illum_N256.npz?rlkey=tj62osr1elh1jo3zulnkv4rwj&dl=1"),
    # --- data: full seismic training and evaluation archives (large) ---
    "data/seismic/dataset_eval.h5":
        ("data", "https://www.dropbox.com/scl/fi/rlt4791uwwref6wybxxal/seismic_eval_lam3.5_v2.h5?rlkey=opo1f8unebxgeqtwkflr2mmvh&dl=1"),
    "data/seismic/dataset_train.h5":
        ("data", "https://www.dropbox.com/scl/fi/j2rkgfl9wa9f53hmo4cts/seismic_train_lam3.5_v2.h5?rlkey=cntcpqsy1cjfdebhcrkw3mm83&dl=1"),
    # --- data: result caches the figures read ---
    "results/darcy_pcn_curated_s0.npz":
        ("data", "https://www.dropbox.com/scl/fi/d0xh30ebgjd57hyj46efa/darcy_pcn_curated_s0.npz?rlkey=wm2qqqbdmibym4w6fmv11tsck&dl=1"),
    "results/darcy_pcn_curated_s1.npz":
        ("data", "https://www.dropbox.com/scl/fi/5t0rf00mloin1xxe55w4k/darcy_pcn_curated_s1.npz?rlkey=f031dacduwk9lbauqpsh8e7di&dl=1"),
    "results/darcy_pcn_curated_s2.npz":
        ("data", "https://www.dropbox.com/scl/fi/0mikvv8oitolmqwlpxp2m/darcy_pcn_curated_s2.npz?rlkey=aswk131cbrch0kxoifo8d2fxh&dl=1"),
    "results/darcy_pcn_oracle_s0.npz":
        ("data", "https://www.dropbox.com/scl/fi/0paqfm4974je5zszvdeq3/darcy_pcn_oracle_s0.npz?rlkey=ndkglcvlau51anyzmyvqa1ahw&dl=1"),
    "results/darcy_pcn_oracle_s1.npz":
        ("data", "https://www.dropbox.com/scl/fi/nr5oxbcyvugkpf0vyp7ua/darcy_pcn_oracle_s1.npz?rlkey=be0h3kss1mimg02zhx6bkzwgf&dl=1"),
    "results/darcy_pcn_oracle_s2.npz":
        ("data", "https://www.dropbox.com/scl/fi/8dgp4rmcpawymxbqufrz7/darcy_pcn_oracle_s2.npz?rlkey=7js1x39n0zedhsg2qzqvtp70r&dl=1"),
    "results/darcy_pcn_single.npz":
        ("data", "https://www.dropbox.com/scl/fi/osmsymemszz11tw769o16/darcy_pcn_single.npz?rlkey=0026e17bnr5nvcnnicstnmj7n&dl=1"),
    "results/seismic_data_kappa.npz":
        ("data", "https://www.dropbox.com/scl/fi/p5c6njj8i1rq2kco7gndh/seismic_data_kappa.npz?rlkey=kj9u41r3os76wphh3t73ke8ff&dl=1"),
    "results/seismic_dps_recon.npz":
        ("data", "https://www.dropbox.com/scl/fi/god0e292w4guoxwk8hk9i/seismic_dps_recon.npz?rlkey=o0c01ngqoe9snp0uweayn3gex&dl=1"),
    "results/seismic_illum_incident.npz":
        ("data", "https://www.dropbox.com/scl/fi/piac32w4ztr20a05phw7k/seismic_illum_incident.npz?rlkey=xa506r7lomn92vjfijtjcqplt&dl=1"),
    "results/seismic_prior_samples.npz":
        ("data", "https://www.dropbox.com/scl/fi/4gcvvox28o1oeduaxx8yb/seismic_prior_samples.npz?rlkey=716o52842trr6upmaoexqbory&dl=1"),
    "results/seismic_prior_seed_coords.npz":
        ("data", "https://www.dropbox.com/scl/fi/8piijy1xfstfv8n1bx4wh/seismic_prior_seed_coords.npz?rlkey=x2613mgky43hpgil1m4c469hl&dl=1"),
    # --- checkpoints ---
    "data/checkpoints/darcy_flow_curated_seed0.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/sxbdinnqh7j7p53qwggn1/darcy_flow_curated_seed0.pth?rlkey=24eg54e5ez7h6kd77jqixkdx5&dl=1"),
    "data/checkpoints/darcy_flow_curated_seed1.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/gyvrzcrn4ddtynue7n3cr/darcy_flow_curated_seed1.pth?rlkey=wkdmvrvam9syl2xgv2sxmumsj&dl=1"),
    "data/checkpoints/darcy_flow_curated_seed2.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/ioxtnnez3fdesvhfgt63f/darcy_flow_curated_seed2.pth?rlkey=7i8t1s8xozr3nyi0cmqdegmhr&dl=1"),
    "data/checkpoints/darcy_flow_oracle_seed0.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/lmhqqiql427v9notm0ezb/darcy_flow_oracle_seed0.pth?rlkey=l1l13124fo3eixbo396qqiu1y&dl=1"),
    "data/checkpoints/darcy_flow_oracle_seed1.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/usw1b7lawix0hrxrzbmov/darcy_flow_oracle_seed1.pth?rlkey=it8at4vkroaf564eujwwp9pml&dl=1"),
    "data/checkpoints/darcy_flow_oracle_seed2.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/jkhqevjuguzfgwbijmmrm/darcy_flow_oracle_seed2.pth?rlkey=820zcrgpviuo24y08mz19ahao&dl=1"),
    "data/checkpoints/seismic_prior_curated_seed0.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/zwhgmpn8obzs0t570ur1b/seismic_prior_curated_seed0.pth?rlkey=qemmopl7w5ycma8oyw1lr1kjw&dl=1"),
    "data/checkpoints/seismic_prior_curated_seed1.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/1nv9n31fpmog8x9834vmu/seismic_prior_curated_seed1.pth?rlkey=t3iwnhypxbxmelvxjujy0ccl9&dl=1"),
    "data/checkpoints/seismic_prior_curated_seed2.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/nz60y65nrddn82c79xvr4/seismic_prior_curated_seed2.pth?rlkey=wexnr03u6g860wa9cq377ts0c&dl=1"),
    "data/checkpoints/seismic_prior_oracle_seed0.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/7stdbj7e6w26pwfyhk5xa/seismic_prior_oracle_seed0.pth?rlkey=w8rz9i7ttg7clizj73zmc0o0r&dl=1"),
    "data/checkpoints/seismic_prior_oracle_seed1.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/4rq339qrzlnqs7or1z3vc/seismic_prior_oracle_seed1.pth?rlkey=33f9169kjtyp9i2s67nc6dexs&dl=1"),
    "data/checkpoints/seismic_prior_oracle_seed2.pth":
        ("checkpoints", "https://www.dropbox.com/scl/fi/dfaszv1f39v7j5b8stie7/seismic_prior_oracle_seed2.pth?rlkey=u03sja42ipogdez9190g0unhs&dl=1"),
}


def _hook(block: int, size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(block * size, total)
    scale, unit = (1e6, "MB") if total >= 1e6 else (1e3, "kB")
    sys.stdout.write(f"\r[resolvability] downloading {done / scale:7.1f} / "
                     f"{total / scale:.1f} {unit} ({100.0 * done / total:5.1f}%)")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


def ensure(*rel_paths: str) -> str | tuple[str, ...]:
    """Return absolute paths for ``rel_paths``, downloading any that are missing.

    Args:
        *rel_paths: repo-relative paths, e.g. ``"results/seismic_dps_recon.npz"``.

    Returns:
        The absolute path if one argument was given, else a tuple of them.

    Raises:
        RuntimeError: a file is missing and is not a registered release artifact, or the
            download did not return the expected file.
    """
    out = tuple(_ensure_one(p) for p in rel_paths)
    return out[0] if len(out) == 1 else out


def _ensure_one(rel: str) -> str:
    path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
    if os.path.exists(path):
        return path

    entry = REGISTRY.get(rel)
    if entry is None:
        raise RuntimeError(
            f"{rel} is missing and is not a released artifact. Regenerate it with the "
            f"producer named for it in the README's pipeline tables.")
    _, url = entry

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[resolvability] {rel} not found; fetching from the public mirror (once).")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, _hook)
        # .npz is a zip and .h5/.pth have their own magic; an expired link returns HTML.
        with open(tmp, "rb") as f:
            head = f.read(8)
        if head[:1] == b"<":
            raise RuntimeError(f"the link for {rel} returned a web page, not the file; "
                               f"see the dataset section of the README.")
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    sz = os.path.getsize(path)
    size = f"{sz / 1e6:.1f} MB" if sz >= 1e6 else f"{sz / 1e3:.0f} kB"
    print(f"[resolvability] saved {rel} ({size})")
    return path


def ensure_tier(tier: str) -> None:
    """Download every artifact in one tier ahead of time (see the module docstring)."""
    names = [k for k, (t, _) in REGISTRY.items() if t == tier]
    if not names:
        raise RuntimeError(f"unknown tier {tier!r}; known tiers: "
                           f"{sorted({t for t, _ in REGISTRY.values()})}")
    for n in sorted(names):
        _ensure_one(n)
