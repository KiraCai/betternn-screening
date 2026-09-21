"""Molecular fingerprint generation: Morgan + AtomPair + 15 RDKit descriptors."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import NamedTuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

MORGAN_BITS = 2048
ATOMPAIR_BITS = 2048
MORGAN_RADIUS = 3
N_DESCRIPTORS = 15

_DESC_FNS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.NumAromaticRings,
    Descriptors.HeavyAtomCount, Descriptors.RingCount,
    Descriptors.FractionCSP3, Descriptors.NumAliphaticRings,
    Descriptors.LabuteASA, Descriptors.BalabanJ,
    Descriptors.BertzCT, Descriptors.MolMR,
]


class MolFeatures(NamedTuple):
    """Features for a single molecule (None canonical_smi signals failure)."""
    canonical_smi: str | None
    morgan: np.ndarray
    atompair: np.ndarray
    descriptors: np.ndarray


class BatchFeatures(NamedTuple):
    """Batch featurization results as aligned numpy arrays."""
    canonical_smiles: np.ndarray  # (n,) object
    morgan: np.ndarray            # (n, 2048) float32
    atompair: np.ndarray          # (n, 2048) float32
    descriptors: np.ndarray       # (n, 15) float32
    valid: np.ndarray             # (n,) bool


def featurize_one(smi: str) -> MolFeatures:
    """Compute Morgan, AtomPair, and 15 descriptor features for one SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return MolFeatures(
            None,
            np.zeros(MORGAN_BITS, dtype=np.float32),
            np.zeros(ATOMPAIR_BITS, dtype=np.float32),
            np.zeros(N_DESCRIPTORS, dtype=np.float32),
        )
    canon = Chem.MolToSmiles(mol, canonical=True)
    morgan_arr = np.zeros(MORGAN_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS),
        morgan_arr,
    )
    atompair_arr = np.zeros(ATOMPAIR_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=ATOMPAIR_BITS),
        atompair_arr,
    )
    desc = np.array([fn(mol) for fn in _DESC_FNS], dtype=np.float32)
    return MolFeatures(canon, morgan_arr, atompair_arr, desc)


def _featurize_chunk(args: tuple[int, list[str]]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker: featurize a contiguous chunk of SMILES."""
    start, smiles_list = args
    n = len(smiles_list)
    canon = np.empty(n, dtype=object)
    morgan = np.zeros((n, MORGAN_BITS), dtype=np.float32)
    atompair = np.zeros((n, ATOMPAIR_BITS), dtype=np.float32)
    desc = np.zeros((n, N_DESCRIPTORS), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, smi in enumerate(smiles_list):
        feat = featurize_one(smi)
        if feat.canonical_smi is not None:
            canon[i] = feat.canonical_smi
            morgan[i] = feat.morgan
            atompair[i] = feat.atompair
            desc[i] = feat.descriptors
            valid[i] = True
        else:
            canon[i] = ""
    return start, canon, morgan, atompair, desc, valid


def featurize_batch(
    smiles_list: list[str],
    n_workers: int = 4,
    chunk_size: int = 10_000,
) -> BatchFeatures:
    """Featurize a list of SMILES in parallel."""
    n = len(smiles_list)
    canon = np.empty(n, dtype=object)
    morgan = np.zeros((n, MORGAN_BITS), dtype=np.float32)
    atompair = np.zeros((n, ATOMPAIR_BITS), dtype=np.float32)
    desc = np.zeros((n, N_DESCRIPTORS), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    chunks = [
        (start, smiles_list[start : min(start + chunk_size, n)])
        for start in range(0, n, chunk_size)
    ]

    with mp.Pool(processes=n_workers) as pool:
        for start, c, m, a, d, v in pool.imap_unordered(_featurize_chunk, chunks):
            end = start + len(c)
            canon[start:end] = c
            morgan[start:end] = m
            atompair[start:end] = a
            desc[start:end] = d
            valid[start:end] = v

    return BatchFeatures(canon, morgan, atompair, desc, valid)


def _load_array(fp_dir: Path, name: str, n_bits: int) -> np.ndarray:
    """Load a fingerprint array as float32, transparently unpacking bit-packed
    storage. Prefers `<name>.packed.npy` (uint8, np.packbits) > `.npz` > `.npy`."""
    packed = fp_dir / f"{name}.packed.npy"
    if packed.exists():
        bits = np.unpackbits(np.load(packed), axis=1)[:, :n_bits]
        return bits.astype(np.float32)
    npz = fp_dir / f"{name}.npz"
    npy = fp_dir / f"{name}.npy"
    if npz.exists():
        return np.load(npz)["data"].astype(np.float32)
    return np.load(npy).astype(np.float32)


def save_fingerprints(fp_dir: str | Path, bf: BatchFeatures, packed: bool = False) -> None:
    """Save a BatchFeatures to disk. With packed=True the Morgan/AtomPair bit
    vectors are stored bit-packed (np.packbits, uint8) — ~32x smaller than the
    float32 .npz. Descriptors/valid/canonical are always plain .npy."""
    fp_dir = Path(fp_dir); fp_dir.mkdir(parents=True, exist_ok=True)
    if packed:
        np.save(fp_dir / "morgan_2048.packed.npy", np.packbits(bf.morgan.astype(np.uint8), axis=1))
        np.save(fp_dir / "atompair.packed.npy", np.packbits(bf.atompair.astype(np.uint8), axis=1))
    else:
        np.savez_compressed(fp_dir / "morgan_2048.npz", data=bf.morgan)
        np.savez_compressed(fp_dir / "atompair.npz", data=bf.atompair)
    np.save(fp_dir / "descriptors.npy", bf.descriptors)
    np.save(fp_dir / "valid.npy", bf.valid)
    np.save(fp_dir / "canonical_smiles.npy", bf.canonical_smiles)


def load_fingerprints(fp_dir: str | Path) -> BatchFeatures:
    """Load pre-computed fingerprints from a directory of .npy/.npz/.packed.npy files."""
    fp_dir = Path(fp_dir)
    morgan = _load_array(fp_dir, "morgan_2048", MORGAN_BITS)
    atompair = _load_array(fp_dir, "atompair", ATOMPAIR_BITS)
    desc_path = fp_dir / "descriptors.npy"
    descriptors = np.load(desc_path) if desc_path.exists() else np.zeros((len(morgan), N_DESCRIPTORS), dtype=np.float32)
    valid_path = fp_dir / "valid.npy"
    valid = np.load(valid_path) if valid_path.exists() else np.ones(len(morgan), dtype=bool)
    canon_path = fp_dir / "canonical_smiles.npy"
    canonical_smiles = np.load(canon_path, allow_pickle=True) if canon_path.exists() else np.empty(len(valid), dtype=object)
    return BatchFeatures(canonical_smiles, morgan, atompair, descriptors, valid)


def concat_fingerprints(morgan: np.ndarray, atompair: np.ndarray) -> np.ndarray:
    """Concatenate Morgan and AtomPair into a 4096-bit input vector."""
    return np.hstack([morgan, atompair]).astype(np.float32)
