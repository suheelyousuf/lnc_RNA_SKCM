import os
import io
import gzip
import json
import logging
import hashlib
import requests
import itertools
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Bio import Entrez, SeqIO
import torch
from torch_geometric.data import HeteroData
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger('HeteroGraphBuilder')
print('start data script')
cc = 2
if False:
    print('eng temp 1')
    print('eng temp 2')

'''
tmp_a = 0
for i in range(3):
    tmp_a += i
print('tmp data test', tmp_a)
'''

# print('data step started')
# print('data step done')



class Config:
    

    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CACHE_DIR: Path = DATA_DIR / "cache"

    ENTREZ_EMAIL: str = "your_email@institution.edu"  # REQUIRED: set to valid email
    ENTREZ_API_KEY: Optional[str] = os.environ.get("NCBI_API_KEY", None)

    SPECIES: str = "Homo sapiens"
    TAXON_ID: int = 9606

    STRING_VERSION: str = "12.0"
    STRING_BASE_URL: str = "https://stringdb-downloads.org/download"
    STRING_PPI_FILE: str = "protein.links.v12.0.txt.gz"
    STRING_MIN_SCORE: int = 700

    HMDD_URL: str = "https://www.cuilab.cn/static/hmdd3/data/alldata_v4.txt"

    LNCRNADISEASE_URL: str = (
        "http://www.rnanut.net/lncrnadisease/resource/data/"
        "experimentally_supported_associations.xlsx"
    )

    MNDR_URL: str = (
        "http://www.rna-society.org/mndr/download/"
        "all-mndr-lncRNA-disease.txt"
    )

    MIRCODE_URL: str = "http://www.mircode.org/download/mircode_highconsfamilies.txt.gz"

    ENCORI_URL: str = (
        "https://rnasysu.com/encori/api/lncRNA"
        "?assembly=hg38&geneType=lncRNA&targetType=miRNA"
    )

    HGNC_DOWNLOAD_URL: str = (
        "https://storage.googleapis.com/public-download-files/"
        "hgnc/tsv/tsv/hgnc_complete_set.txt"
    )

    DOID_OBO_URL: str = (
        "https://raw.githubusercontent.com/DiseaseOntology/"
        "HumanDiseaseOntology/main/src/ontology/doid.obo"
    )

    KMER_SIZES: List[int] = [3, 4]
    NEGATIVE_SAMPLE_RATIO: float = 1.0  # ratio of neg to pos samples
    RANDOM_SEED: int = 42

    TARGET_DISEASE: str = "melanoma"
    MESH_MELANOMA: str = "D008545"
    DOID_MELANOMA: str = "DOID:1909"



def ensure_dirs():
    
    for d in [Config.DATA_DIR, Config.RAW_DIR, Config.PROCESSED_DIR, Config.CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def cached_download(url: str, filename: str, force: bool = False) -> Path:
    
    ensure_dirs()
    filepath = Config.RAW_DIR / filename
    if filepath.exists() and not force:
        logger.info(f"Using cached file: {filepath}")
        return filepath

    logger.info(f"Downloading: {url}")
    headers = {"User-Agent": "HeteroGraphBuilder/1.0 (Research; Bioinformatics)"}
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.SSLError:
        logger.warning(f"SSL verification failed for {url}, retrying without verify.")
        resp = requests.get(url, headers=headers, stream=True, timeout=120, verify=False)
        resp.raise_for_status()

    with open(filepath, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    logger.info(f"Saved to: {filepath}")
    return filepath


def safe_request_json(url: str, params: Optional[dict] = None) -> dict:
    
    headers = {"User-Agent": "HeteroGraphBuilder/1.0 (Research; Bioinformatics)"}
    resp = requests.get(url, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()



class GeneNormalizer:
    

    def __init__(self):
        self.symbol_to_hgnc: Dict[str, str] = {}
        self.alias_to_hgnc: Dict[str, str] = {}
        self.hgnc_to_entrez: Dict[str, str] = {}
        self.entrez_to_hgnc: Dict[str, str] = {}
        self._loaded = False

    def load(self, force_download: bool = False):
        
        if self._loaded:
            return

        filepath = cached_download(
            Config.HGNC_DOWNLOAD_URL, "hgnc_complete_set.txt", force=force_download
        )
        logger.info("Parsing HGNC gene nomenclature...")
        df = pd.read_csv(filepath, sep="\t", low_memory=False)

        for _, row in df.iterrows():
            symbol = str(row.get("symbol", "")).strip()
            if not symbol:
                continue

            self.symbol_to_hgnc[symbol.upper()] = symbol

            prev = str(row.get("prev_symbol", ""))
            if prev and prev != "nan":
                for alias in prev.split("|"):
                    alias = alias.strip().upper()
                    if alias:
                        self.alias_to_hgnc[alias] = symbol

            aliases = str(row.get("alias_symbol", ""))
            if aliases and aliases != "nan":
                for alias in aliases.split("|"):
                    alias = alias.strip().upper()
                    if alias:
                        self.alias_to_hgnc[alias] = symbol

            entrez = str(row.get("entrez_id", "")).strip()
            if entrez and entrez != "nan":
                self.hgnc_to_entrez[symbol] = entrez
                self.entrez_to_hgnc[entrez] = symbol

        self._loaded = True
        logger.info(
            f"HGNC loaded: {len(self.symbol_to_hgnc)} symbols, "
            f"{len(self.alias_to_hgnc)} aliases, "
            f"{len(self.hgnc_to_entrez)} Entrez mappings"
        )

    def normalize(self, name: str) -> Optional[str]:
        
        if not name:
            return None
        name_upper = name.strip().upper()

        if name_upper in self.symbol_to_hgnc:
            return self.symbol_to_hgnc[name_upper]
        if name_upper in self.alias_to_hgnc:
            return self.alias_to_hgnc[name_upper]
        return None

    def get_entrez(self, hgnc_symbol: str) -> Optional[str]:
        
        return self.hgnc_to_entrez.get(hgnc_symbol)



class LncRNADiseaseParser:
    

    @staticmethod
    def parse_lncrnadisease(normalizer: GeneNormalizer) -> pd.DataFrame:
        
        logger.info("Fetching LncRNADisease v3.0 associations...")

        try:
            filepath = cached_download(
                Config.LNCRNADISEASE_URL, "lncrnadisease_v3_exp.xlsx"
            )
            df = pd.read_excel(filepath)
        except Exception as e:
            logger.warning(
                f"LncRNADisease download failed ({e}). "
                "Creating placeholder structure. Manual download required from: "
                "http://www.rnanut.net/lncrnadisease/"
            )
            return pd.DataFrame(columns=["lncrna", "disease", "source", "confidence"])

        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if "lncrna" in col_lower or "rna_name" in col_lower:
                col_map[col] = "lncrna_raw"
            elif "disease" in col_lower:
                col_map[col] = "disease"
            elif "species" in col_lower or "organism" in col_lower:
                col_map[col] = "species"
            elif "confidence" in col_lower or "score" in col_lower:
                col_map[col] = "confidence"
        df = df.rename(columns=col_map)

        if "species" in df.columns:
            df = df[
                df["species"].str.contains("Homo sapiens|Human|9606", 
                                           case=False, na=False)
            ]

        records = []
        for _, row in df.iterrows():
            lncrna_raw = str(row.get("lncrna_raw", "")).strip()
            disease = str(row.get("disease", "")).strip().lower()
            confidence = row.get("confidence", 1.0)

            normalized = normalizer.normalize(lncrna_raw)
            lncrna = normalized if normalized else lncrna_raw.upper()

            if lncrna and disease:
                records.append({
                    "lncrna": lncrna,
                    "disease": disease,
                    "source": "LncRNADisease_v3",
                    "confidence": float(confidence) if pd.notna(confidence) else 1.0,
                })

        result = pd.DataFrame(records)
        logger.info(f"LncRNADisease v3.0: {len(result)} human associations parsed")
        return result

    @staticmethod
    def parse_mndr(normalizer: GeneNormalizer) -> pd.DataFrame:
        
        logger.info("Fetching MNDR v4.0 associations...")

        try:
            filepath = cached_download(Config.MNDR_URL, "mndr_v4_lncrna_disease.txt")
            with open(filepath, "r", errors="ignore") as _fh:
                first_line = _fh.readline()
            if first_line.strip().startswith("<"):
                logger.warning(
                    "MNDR returned HTML instead of data. "
                    "Manual download required from: http://www.rna-society.org/mndr/"
                )
                filepath.unlink(missing_ok=True)
                return pd.DataFrame(columns=["lncrna", "disease", "source", "confidence"])
            df = pd.read_csv(filepath, sep="\t", header=0)
        except Exception as e:
            logger.warning(
                f"MNDR download failed ({e}). "
                "Manual download from: http://www.rna-society.org/mndr/"
            )
            return pd.DataFrame(columns=["lncrna", "disease", "source", "confidence"])

        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if "rna" in cl and "symbol" in cl or "ncrna" in cl:
                col_map[col] = "lncrna_raw"
            elif "disease" in cl:
                col_map[col] = "disease"
            elif "species" in cl:
                col_map[col] = "species"
            elif "score" in cl or "confidence" in cl:
                col_map[col] = "confidence"
            elif "type" in cl and "rna" in cl:
                col_map[col] = "rna_type"
        df = df.rename(columns=col_map)

        if "species" in df.columns:
            df = df[df["species"].str.contains("Homo sapiens|Human", case=False, na=False)]
        if "rna_type" in df.columns:
            df = df[df["rna_type"].str.contains("lncRNA|lnc", case=False, na=False)]

        records = []
        for _, row in df.iterrows():
            lncrna_raw = str(row.get("lncrna_raw", "")).strip()
            disease = str(row.get("disease", "")).strip().lower()
            confidence = row.get("confidence", 1.0)

            normalized = normalizer.normalize(lncrna_raw)
            lncrna = normalized if normalized else lncrna_raw.upper()

            if lncrna and disease:
                records.append({
                    "lncrna": lncrna,
                    "disease": disease,
                    "source": "MNDR_v4",
                    "confidence": float(confidence) if pd.notna(confidence) else 1.0,
                })

        result = pd.DataFrame(records)
        logger.info(f"MNDR v4.0: {len(result)} human lncRNA-disease associations parsed")
        return result


class HMDDParser:
    

    @staticmethod
    def parse(normalizer: GeneNormalizer) -> pd.DataFrame:
        
        logger.info("Fetching HMDD v4.0 miRNA-disease associations...")

        try:
            filepath = cached_download(Config.HMDD_URL, "hmdd_v4_alldata.txt")
            df = pd.read_csv(filepath, sep="\t", header=0)
        except Exception as e:
            logger.warning(f"HMDD download failed ({e}). Manual download required.")
            return pd.DataFrame(columns=["mirna", "disease", "source", "confidence"])

        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if "mir" in cl and "disease" not in cl:
                col_map[col] = "mirna_raw"
            elif "disease" in cl:
                col_map[col] = "disease"
            elif "category" in cl:
                col_map[col] = "category"
        df = df.rename(columns=col_map)

        records = []
        for _, row in df.iterrows():
            mirna = str(row.get("mirna_raw", "")).strip().lower()
            disease = str(row.get("disease", "")).strip().lower()

            if mirna.startswith("hsa-"):
                mirna = mirna  # keep human miRNAs
            else:
                continue  # skip non-human

            if mirna and disease:
                records.append({
                    "mirna": mirna,
                    "disease": disease,
                    "source": "HMDD_v4",
                    "confidence": 1.0,
                })

        result = pd.DataFrame(records)
        logger.info(f"HMDD v4.0: {len(result)} human miRNA-disease associations parsed")
        return result


class LncRNAMiRNAParser:
    

    @staticmethod
    def parse_mircode(normalizer: GeneNormalizer) -> pd.DataFrame:
        
        logger.info("Fetching miRCode lncRNA-miRNA interactions...")

        try:
            filepath = cached_download(
                Config.MIRCODE_URL, "mircode_highconsfamilies.txt.gz"
            )
            df = pd.read_csv(filepath, sep="\t", compression="gzip")
        except Exception as e:
            logger.warning(f"miRCode download failed ({e}).")
            return pd.DataFrame(columns=["lncrna", "mirna", "source", "confidence"])

        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if "gene" in cl and "class" not in cl and "type" not in cl:
                col_map[col] = "gene"
            elif "class" in cl or "type" in cl or "biotype" in cl:
                col_map[col] = "gene_type"
            elif "mir" in cl and "family" in cl:
                col_map[col] = "mirna_family"
            elif "mir" in cl:
                col_map[col] = "mirna_raw"
        df = df.rename(columns=col_map)

        if "gene_type" in df.columns:
            df = df[df["gene_type"].str.contains("lncRNA|lincRNA|antisense",
                                                  case=False, na=False)]

        records = []
        mirna_col = "mirna_raw" if "mirna_raw" in df.columns else "mirna_family"
        for _, row in df.iterrows():
            gene = str(row.get("gene", "")).strip()
            mirna = str(row.get(mirna_col, "")).strip().lower()

            normalized = normalizer.normalize(gene)
            lncrna = normalized if normalized else gene.upper()

            if lncrna and mirna:
                records.append({
                    "lncrna": lncrna,
                    "mirna": mirna,
                    "source": "miRCode",
                    "confidence": 1.0,
                })

        result = pd.DataFrame(records).drop_duplicates(subset=["lncrna", "mirna"])
        logger.info(f"miRCode: {len(result)} lncRNA-miRNA interactions parsed")
        return result

    @staticmethod
    def parse_encori(normalizer: GeneNormalizer) -> pd.DataFrame:
        
        logger.info("Fetching ENCORI/StarBase lncRNA-miRNA interactions...")

        try:
            data = safe_request_json(Config.ENCORI_URL)
        except Exception as e:
            logger.warning(
                f"ENCORI API failed ({e}). Falling back to manual download. "
                "Visit: https://rnasysu.com/encori/"
            )
            return pd.DataFrame(columns=["lncrna", "mirna", "source", "confidence"])

        records = []
        if isinstance(data, list):
            for entry in data:
                lncrna_raw = entry.get("geneName", "")
                mirna = entry.get("miRNAname", "").lower()
                clip_reads = entry.get("clipExpNum", 0)

                normalized = normalizer.normalize(lncrna_raw)
                lncrna = normalized if normalized else lncrna_raw.upper()

                if lncrna and mirna and mirna.startswith("hsa-"):
                    records.append({
                        "lncrna": lncrna,
                        "mirna": mirna,
                        "source": "ENCORI",
                        "confidence": min(int(clip_reads) / 10.0, 1.0),
                    })

        result = pd.DataFrame(records).drop_duplicates(subset=["lncrna", "mirna"])
        logger.info(f"ENCORI: {len(result)} lncRNA-miRNA interactions parsed")
        return result


class STRINGParser:
    

    @staticmethod
    def parse(normalizer: GeneNormalizer) -> pd.DataFrame:
        
        logger.info("Fetching STRING PPI data (human, score > 700)...")

        string_url = (
            f"{Config.STRING_BASE_URL}/"
            f"protein.links.v{Config.STRING_VERSION}/"
            f"{Config.TAXON_ID}.protein.links.v{Config.STRING_VERSION}.txt.gz"
        )

        try:
            filepath = cached_download(
                string_url, f"string_{Config.TAXON_ID}_links.txt.gz"
            )
        except Exception as e:
            logger.warning(f"STRING download failed ({e}). File is large (~400MB).")
            return pd.DataFrame(
                columns=["protein_a", "protein_b", "combined_score", "source"]
            )

        logger.info("Parsing STRING interactions (this may take several minutes)...")
        records = []
        chunk_size = 500_000
        taxon_prefix = f"{Config.TAXON_ID}."

        reader = pd.read_csv(
            filepath, sep=" ", compression="gzip", chunksize=chunk_size
        )

        for chunk in reader:
            high_conf = chunk[chunk["combined_score"] >= Config.STRING_MIN_SCORE]

            for _, row in high_conf.iterrows():
                prot_a = str(row["protein1"]).replace(taxon_prefix, "")
                prot_b = str(row["protein2"]).replace(taxon_prefix, "")
                score = int(row["combined_score"])
                records.append({
                    "protein_a": prot_a,
                    "protein_b": prot_b,
                    "combined_score": score,
                    "source": "STRING",
                })

        result = pd.DataFrame(records)
        logger.info(f"STRING: {len(result)} PPIs with score >= {Config.STRING_MIN_SCORE}")

        result = STRINGParser._map_ensembl_to_symbol(result, normalizer)
        return result

    @staticmethod
    def _map_ensembl_to_symbol(
        df: pd.DataFrame, normalizer: GeneNormalizer
    ) -> pd.DataFrame:
        
        alias_url = (
            f"{Config.STRING_BASE_URL}/"
            f"protein.aliases.v{Config.STRING_VERSION}/"
            f"{Config.TAXON_ID}.protein.aliases.v{Config.STRING_VERSION}.txt.gz"
        )
        try:
            alias_path = cached_download(
                alias_url, f"string_{Config.TAXON_ID}_aliases.txt.gz"
            )
            aliases_df = pd.read_csv(alias_path, sep="\t", compression="gzip")
            ensp_to_symbol = {}
            hugo_aliases = aliases_df[
                aliases_df["source"].str.contains("BioMart_HUGO", na=False)
            ]
            for _, row in hugo_aliases.iterrows():
                ensp = str(row.iloc[0]).replace(f"{Config.TAXON_ID}.", "")
                symbol = str(row.iloc[1]).strip()
                ensp_to_symbol[ensp] = symbol

            df = df.copy()
            df["gene_a"] = df["protein_a"].map(ensp_to_symbol)
            df["gene_b"] = df["protein_b"].map(ensp_to_symbol)
            df = df.dropna(subset=["gene_a", "gene_b"])

            df["gene_a"] = df["gene_a"].apply(
                lambda x: normalizer.normalize(x) or x.upper()
            )
            df["gene_b"] = df["gene_b"].apply(
                lambda x: normalizer.normalize(x) or x.upper()
            )

        except Exception as e:
            logger.warning(f"STRING alias mapping failed ({e}). Using raw IDs.")
            df["gene_a"] = df["protein_a"]
            df["gene_b"] = df["protein_b"]

        return df


class DiseaseOntologyParser:
    

    @staticmethod
    def parse_doid() -> Dict[str, dict]:
        
        logger.info("Fetching Disease Ontology (DOID) OBO file...")

        try:
            filepath = cached_download(Config.DOID_OBO_URL, "doid.obo")
        except Exception as e:
            logger.warning(f"DOID download failed ({e}).")
            return {}

        ontology = {}
        current_id = None
        current_entry = {"name": "", "parents": [], "mesh_xrefs": [], "alt_ids": []}

        with open(filepath, "r") as fh:
            for line in fh:
                line = line.strip()
                if line == "[Term]":
                    if current_id:
                        ontology[current_id] = current_entry
                    current_id = None
                    current_entry = {
                        "name": "", "parents": [], "mesh_xrefs": [], "alt_ids": []
                    }
                elif line.startswith("id: DOID:"):
                    current_id = line.split("id: ")[1]
                elif line.startswith("name: "):
                    current_entry["name"] = line.split("name: ")[1].lower()
                elif line.startswith("is_a: DOID:"):
                    parent = line.split("is_a: ")[1].split(" !")[0]
                    current_entry["parents"].append(parent)
                elif line.startswith("xref: MESH:"):
                    mesh = line.split("xref: ")[1]
                    current_entry["mesh_xrefs"].append(mesh)

            if current_id:
                ontology[current_id] = current_entry

        logger.info(f"DOID: {len(ontology)} disease terms parsed")
        return ontology

    @staticmethod
    def compute_semantic_similarity(
        ontology: Dict[str, dict], disease_set: Set[str], top_k: int = 50
    ) -> pd.DataFrame:
        
        logger.info("Computing disease-disease semantic similarities...")

        name_to_doid = {}
        for doid, info in ontology.items():
            name_to_doid[info["name"]] = doid

        dag = nx.DiGraph()
        for doid, info in ontology.items():
            for parent in info["parents"]:
                dag.add_edge(doid, parent)  # child → parent

        total_terms = len(ontology)
        ic_values = {}
        for doid in ontology:
            descendants = nx.ancestors(dag, doid) if dag.has_node(doid) else set()
            desc_count = len(descendants) + 1  # include self
            ic_values[doid] = -np.log(desc_count / total_terms)

        def get_ancestors(doid: str) -> Set[str]:
            
            if not dag.has_node(doid):
                return set()
            return nx.descendants(dag, doid) | {doid}  # descendants in DAG = ancestors

        def lin_similarity(doid_a: str, doid_b: str) -> float:
            
            if doid_a == doid_b:
                return 1.0
            ancestors_a = get_ancestors(doid_a)
            ancestors_b = get_ancestors(doid_b)
            common = ancestors_a & ancestors_b
            if not common:
                return 0.0
            max_ic = max(ic_values.get(c, 0) for c in common)
            ic_a = ic_values.get(doid_a, 0)
            ic_b = ic_values.get(doid_b, 0)
            denom = ic_a + ic_b
            if denom == 0:
                return 0.0
            return (2 * max_ic) / denom

        matched_diseases = {}
        for disease_name in disease_set:
            if disease_name in name_to_doid:
                matched_diseases[disease_name] = name_to_doid[disease_name]
            else:
                for doid_name, doid_id in name_to_doid.items():
                    if disease_name in doid_name or doid_name in disease_name:
                        matched_diseases[disease_name] = doid_id
                        break

        disease_list = list(matched_diseases.keys())
        records = []

        for i, d_a in enumerate(disease_list):
            sims = []
            doid_a = matched_diseases[d_a]
            for j, d_b in enumerate(disease_list):
                if i >= j:
                    continue
                doid_b = matched_diseases[d_b]
                sim = lin_similarity(doid_a, doid_b)
                if sim > 0.1:  # threshold to reduce noise
                    sims.append((d_b, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            for d_b, sim in sims[:top_k]:
                records.append({
                    "disease_a": d_a,
                    "disease_b": d_b,
                    "similarity": sim,
                })

        result = pd.DataFrame(records)
        logger.info(f"Disease similarity: {len(result)} pairs computed")
        return result



class KmerFeatureEncoder:
    

    def __init__(self, kmer_sizes: List[int] = None):
        self.kmer_sizes = kmer_sizes or Config.KMER_SIZES
        self.vectorizers: Dict[str, TfidfVectorizer] = {}

    @staticmethod
    def sequence_to_kmers(sequence: str, k: int) -> str:
        
        sequence = sequence.upper().replace("N", "")
        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        return " ".join(kmers)

    def fit_transform(
        self, sequences: Dict[str, str], node_type: str
    ) -> Tuple[np.ndarray, List[str]]:
        
        node_ids = list(sequences.keys())
        if not node_ids:
            return np.zeros((0, 0)), []

        documents = []
        for nid in node_ids:
            seq = sequences[nid]
            kmer_parts = []
            for k in self.kmer_sizes:
                kmer_parts.append(self.sequence_to_kmers(seq, k))
            documents.append(" ".join(kmer_parts))

        vectorizer = TfidfVectorizer(
            analyzer="word",
            lowercase=False,
            max_features=5000,  # Cap dimensionality
            sublinear_tf=True,
            norm="l2",
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        self.vectorizers[node_type] = vectorizer

        logger.info(
            f"K-mer TF-IDF for '{node_type}': "
            f"{tfidf_matrix.shape[0]} nodes × {tfidf_matrix.shape[1]} features"
        )
        return tfidf_matrix.toarray(), node_ids

    @staticmethod
    def fetch_sequences_batch(
        gene_ids: List[str], db: str = "nucleotide", rettype: str = "fasta"
    ) -> Dict[str, str]:
        
        Entrez.email = Config.ENTREZ_EMAIL
        if Config.ENTREZ_API_KEY:
            Entrez.api_key = Config.ENTREZ_API_KEY

        sequences = {}
        batch_size = 200  # NCBI recommends ≤200 per request

        for i in range(0, len(gene_ids), batch_size):
            batch = gene_ids[i:i + batch_size]
            try:
                search_handle = Entrez.esearch(
                    db=db,
                    term=" OR ".join([f"{gid}[Gene Name] AND Homo sapiens[Organism]"
                                      for gid in batch[:10]]),  # limit query size
                    retmax=len(batch),
                )
                search_results = Entrez.read(search_handle)
                search_handle.close()

                id_list = search_results.get("IdList", [])
                if not id_list:
                    continue

                fetch_handle = Entrez.efetch(
                    db=db, id=id_list, rettype=rettype, retmode="text"
                )
                for record in SeqIO.parse(fetch_handle, "fasta"):
                    seq_str = str(record.seq)
                    if len(seq_str) > 100:  # minimum length filter
                        sequences[record.id] = seq_str
                fetch_handle.close()

            except Exception as e:
                logger.warning(f"Entrez fetch failed for batch {i}: {e}")
                continue

        return sequences



class NegativeSampler:
    

    def __init__(self, seed: int = Config.RANDOM_SEED):
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        positive_pairs: Set[Tuple[str, str]],
        all_lncrnas: List[str],
        all_diseases: List[str],
        ratio: float = Config.NEGATIVE_SAMPLE_RATIO,
    ) -> List[Tuple[str, str]]:
        
        n_negatives = int(len(positive_pairs) * ratio)
        logger.info(
            f"Generating {n_negatives} negative samples "
            f"(ratio={ratio}, positives={len(positive_pairs)})"
        )

        lncrna_degrees = Counter(p[0] for p in positive_pairs)
        disease_degrees = Counter(p[1] for p in positive_pairs)

        lncrna_weights = np.array([
            lncrna_degrees.get(l, 1) for l in all_lncrnas
        ], dtype=float)
        lncrna_weights /= lncrna_weights.sum()

        disease_weights = np.array([
            disease_degrees.get(d, 1) for d in all_diseases
        ], dtype=float)
        disease_weights /= disease_weights.sum()

        negatives = set()
        max_attempts = n_negatives * 10

        attempts = 0
        while len(negatives) < n_negatives and attempts < max_attempts:
            l_idx = self.rng.choice(len(all_lncrnas), p=lncrna_weights)
            d_idx = self.rng.choice(len(all_diseases), p=disease_weights)
            pair = (all_lncrnas[l_idx], all_diseases[d_idx])

            if pair not in positive_pairs and pair not in negatives:
                negatives.add(pair)
            attempts += 1

        result = list(negatives)
        logger.info(f"Generated {len(result)} negative samples")
        return result



class HeteroGraphBuilder:
    

    def __init__(self, config: type = Config):
        self.config = config
        self.normalizer = GeneNormalizer()
        self.kmer_encoder = KmerFeatureEncoder()
        self.negative_sampler = NegativeSampler()

        self.lncrna_disease_df: Optional[pd.DataFrame] = None
        self.mirna_disease_df: Optional[pd.DataFrame] = None
        self.lncrna_mirna_df: Optional[pd.DataFrame] = None
        self.ppi_df: Optional[pd.DataFrame] = None
        self.disease_sim_df: Optional[pd.DataFrame] = None

        self.node_indices: Dict[str, Dict[str, int]] = {
            "lncrna": {},
            "mirna": {},
            "mrna": {},
            "disease": {},
        }

        self.G: Optional[nx.Graph] = None

        self.hetero_data: Optional[HeteroData] = None

    def build(self) -> HeteroData:
        
        logger.info("=" * 70)
        logger.info("HETEROGENEOUS GRAPH CONSTRUCTION PIPELINE")
        logger.info("=" * 70)

        ensure_dirs()

        logger.info("\n[Step 1/7] Loading HGNC gene nomenclature...")
        self.normalizer.load()

        logger.info("\n[Step 2/7] Parsing biological data sources...")
        self._parse_all_sources()

        logger.info("\n[Step 3/7] Registering nodes...")
        self._register_nodes()

        logger.info("\n[Step 4/7] Constructing NetworkX heterogeneous graph...")
        self._build_networkx_graph()

        logger.info("\n[Step 5/7] Computing node features (k-mer TF-IDF)...")
        node_features = self._compute_node_features()

        logger.info("\n[Step 6/7] Generating negative samples...")
        negative_pairs = self._generate_negatives()

        logger.info("\n[Step 7/7] Building PyTorch Geometric HeteroData...")
        self.hetero_data = self._build_pyg_heterodata(node_features, negative_pairs)

        self._print_summary()

        return self.hetero_data

    def _parse_all_sources(self):
        
        lrd_df = LncRNADiseaseParser.parse_lncrnadisease(self.normalizer)
        mndr_df = LncRNADiseaseParser.parse_mndr(self.normalizer)
        self.lncrna_disease_df = pd.concat(
            [lrd_df, mndr_df], ignore_index=True
        ).drop_duplicates(subset=["lncrna", "disease"])

        self.mirna_disease_df = HMDDParser.parse(self.normalizer)

        mircode_df = LncRNAMiRNAParser.parse_mircode(self.normalizer)
        encori_df = LncRNAMiRNAParser.parse_encori(self.normalizer)
        self.lncrna_mirna_df = pd.concat(
            [mircode_df, encori_df], ignore_index=True
        ).drop_duplicates(subset=["lncrna", "mirna"])

        self.ppi_df = STRINGParser.parse(self.normalizer)

        ontology = DiseaseOntologyParser.parse_doid()
        all_diseases = set()
        if len(self.lncrna_disease_df) > 0:
            all_diseases.update(self.lncrna_disease_df["disease"].unique())
        if len(self.mirna_disease_df) > 0:
            all_diseases.update(self.mirna_disease_df["disease"].unique())
        self.disease_sim_df = DiseaseOntologyParser.compute_semantic_similarity(
            ontology, all_diseases
        )

    def _register_nodes(self):
        
        lncrna_set = set()
        if len(self.lncrna_disease_df) > 0:
            lncrna_set.update(self.lncrna_disease_df["lncrna"].unique())
        if len(self.lncrna_mirna_df) > 0:
            lncrna_set.update(self.lncrna_mirna_df["lncrna"].unique())
        for i, node in enumerate(sorted(lncrna_set)):
            self.node_indices["lncrna"][node] = i

        mirna_set = set()
        if len(self.mirna_disease_df) > 0:
            mirna_set.update(self.mirna_disease_df["mirna"].unique())
        if len(self.lncrna_mirna_df) > 0:
            mirna_set.update(self.lncrna_mirna_df["mirna"].unique())
        for i, node in enumerate(sorted(mirna_set)):
            self.node_indices["mirna"][node] = i

        mrna_set = set()
        if len(self.ppi_df) > 0:
            if "gene_a" in self.ppi_df.columns:
                mrna_set.update(self.ppi_df["gene_a"].unique())
                mrna_set.update(self.ppi_df["gene_b"].unique())
        for i, node in enumerate(sorted(mrna_set)):
            self.node_indices["mrna"][node] = i

        disease_set = set()
        if len(self.lncrna_disease_df) > 0:
            disease_set.update(self.lncrna_disease_df["disease"].unique())
        if len(self.mirna_disease_df) > 0:
            disease_set.update(self.mirna_disease_df["disease"].unique())
        if len(self.disease_sim_df) > 0:
            disease_set.update(self.disease_sim_df["disease_a"].unique())
            disease_set.update(self.disease_sim_df["disease_b"].unique())
        for i, node in enumerate(sorted(disease_set)):
            self.node_indices["disease"][node] = i

        for ntype, indices in self.node_indices.items():
            logger.info(f"  {ntype}: {len(indices)} nodes")

    def _build_networkx_graph(self):
        
        self.G = nx.Graph()

        for ntype, indices in self.node_indices.items():
            for node_id, idx in indices.items():
                self.G.add_node(
                    f"{ntype}::{node_id}",
                    node_type=ntype,
                    node_idx=idx,
                    label=node_id,
                )

        if len(self.lncrna_disease_df) > 0:
            for _, row in self.lncrna_disease_df.iterrows():
                src = f"lncrna::{row['lncrna']}"
                dst = f"disease::{row['disease']}"
                if self.G.has_node(src) and self.G.has_node(dst):
                    self.G.add_edge(
                        src, dst,
                        edge_type="lncrna_disease",
                        weight=row.get("confidence", 1.0),
                        source=row.get("source", ""),
                    )

        if len(self.mirna_disease_df) > 0:
            for _, row in self.mirna_disease_df.iterrows():
                src = f"mirna::{row['mirna']}"
                dst = f"disease::{row['disease']}"
                if self.G.has_node(src) and self.G.has_node(dst):
                    self.G.add_edge(
                        src, dst,
                        edge_type="mirna_disease",
                        weight=row.get("confidence", 1.0),
                        source=row.get("source", ""),
                    )

        if len(self.lncrna_mirna_df) > 0:
            for _, row in self.lncrna_mirna_df.iterrows():
                src = f"lncrna::{row['lncrna']}"
                dst = f"mirna::{row['mirna']}"
                if self.G.has_node(src) and self.G.has_node(dst):
                    self.G.add_edge(
                        src, dst,
                        edge_type="lncrna_mirna",
                        weight=row.get("confidence", 1.0),
                        source=row.get("source", ""),
                    )

        if len(self.ppi_df) > 0 and "gene_a" in self.ppi_df.columns:
            for _, row in self.ppi_df.iterrows():
                src = f"mrna::{row['gene_a']}"
                dst = f"mrna::{row['gene_b']}"
                if self.G.has_node(src) and self.G.has_node(dst):
                    self.G.add_edge(
                        src, dst,
                        edge_type="ppi",
                        weight=row["combined_score"] / 1000.0,
                        source="STRING",
                    )

        if len(self.disease_sim_df) > 0:
            for _, row in self.disease_sim_df.iterrows():
                src = f"disease::{row['disease_a']}"
                dst = f"disease::{row['disease_b']}"
                if self.G.has_node(src) and self.G.has_node(dst):
                    self.G.add_edge(
                        src, dst,
                        edge_type="disease_similarity",
                        weight=row["similarity"],
                        source="DOID",
                    )

        logger.info(
            f"  NetworkX graph: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )

    def _compute_node_features(self) -> Dict[str, torch.Tensor]:
        
        feature_dim = 256  # Target feature dimension for all node types
        node_features = {}

        for ntype, indices in self.node_indices.items():
            n_nodes = len(indices)
            if n_nodes == 0:
                node_features[ntype] = torch.zeros((0, feature_dim))
                continue

            if ntype in ("lncrna", "mirna", "mrna"):
                sequences = self._fetch_node_sequences(ntype, list(indices.keys()))

                if len(sequences) >= max(10, n_nodes * 0.3):
                    feat_matrix, ordered_ids = self.kmer_encoder.fit_transform(
                        sequences, ntype
                    )
                    full_features = np.zeros((n_nodes, feat_matrix.shape[1]))
                    for seq_idx, node_id in enumerate(ordered_ids):
                        if node_id in indices:
                            full_features[indices[node_id]] = feat_matrix[seq_idx]

                    if full_features.shape[1] > feature_dim:
                        from sklearn.decomposition import TruncatedSVD
                        svd = TruncatedSVD(n_components=feature_dim, random_state=42)
                        full_features = svd.fit_transform(
                            csr_matrix(full_features)
                        )
                    elif full_features.shape[1] < feature_dim:
                        pad = np.zeros((n_nodes, feature_dim - full_features.shape[1]))
                        full_features = np.hstack([full_features, pad])

                    node_features[ntype] = torch.FloatTensor(full_features)
                else:
                    logger.info(
                        f"  Insufficient sequences for {ntype}. "
                        f"Using Xavier initialization ({n_nodes} nodes)."
                    )
                    feat = torch.empty(n_nodes, feature_dim)
                    torch.nn.init.xavier_uniform_(feat)
                    node_features[ntype] = feat

            elif ntype == "disease":
                feat = self._compute_disease_features(indices, feature_dim)
                node_features[ntype] = feat

        return node_features

    def _fetch_node_sequences(
        self, node_type: str, node_ids: List[str]
    ) -> Dict[str, str]:
        
        cache_file = Config.CACHE_DIR / f"{node_type}_sequences.json"

        if cache_file.exists():
            with open(cache_file, "r") as fh:
                cached = json.load(fh)
            logger.info(f"  Loaded {len(cached)} cached sequences for {node_type}")
            return cached

        logger.info(
            f"  Sequence fetch for {node_type}: "
            f"Set ENTREZ_EMAIL in Config and run with network access."
        )

        sequences = {}
        try:
            if node_type == "mirna":
                sequences = self._fetch_mirna_sequences(node_ids[:500])
            else:
                sequences = KmerFeatureEncoder.fetch_sequences_batch(
                    node_ids[:200], db="nucleotide"
                )
        except Exception as e:
            logger.warning(f"  Sequence fetch failed for {node_type}: {e}")

        if sequences:
            with open(cache_file, "w") as fh:
                json.dump(sequences, fh)

        return sequences

    @staticmethod
    def _fetch_mirna_sequences(mirna_ids: List[str]) -> Dict[str, str]:
        
        Entrez.email = Config.ENTREZ_EMAIL
        sequences = {}

        try:
            for mirna_id in mirna_ids[:100]:  # Limit to avoid rate limiting
                search_term = f"{mirna_id}[Title] AND miRNA[Filter]"
                handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=1)
                result = Entrez.read(handle)
                handle.close()

                if result["IdList"]:
                    fetch_handle = Entrez.efetch(
                        db="nucleotide", id=result["IdList"][0],
                        rettype="fasta", retmode="text"
                    )
                    record = SeqIO.read(io.StringIO(fetch_handle.read()), "fasta")
                    sequences[mirna_id] = str(record.seq)
                    fetch_handle.close()
        except Exception as e:
            logger.warning(f"miRBase fetch error: {e}")

        return sequences

    def _compute_disease_features(
        self, disease_indices: Dict[str, int], feature_dim: int
    ) -> torch.Tensor:
        
        n_diseases = len(disease_indices)
        features = np.zeros((n_diseases, feature_dim))

        if len(self.disease_sim_df) > 0:
            sim_dim = min(n_diseases, feature_dim - 1)
            disease_list = sorted(disease_indices.keys())

            for _, row in self.disease_sim_df.iterrows():
                d_a = row["disease_a"]
                d_b = row["disease_b"]
                sim = row["similarity"]
                if d_a in disease_indices and d_b in disease_indices:
                    idx_a = disease_indices[d_a]
                    idx_b = disease_indices[d_b]
                    if idx_b < sim_dim:
                        features[idx_a, idx_b] = sim
                    if idx_a < sim_dim:
                        features[idx_b, idx_a] = sim

        for disease_name, idx in disease_indices.items():
            if Config.TARGET_DISEASE in disease_name:
                features[idx, -1] = 1.0

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms

        return torch.FloatTensor(features)

    def _generate_negatives(self) -> List[Tuple[str, str]]:
        
        positive_pairs = set()
        if len(self.lncrna_disease_df) > 0:
            for _, row in self.lncrna_disease_df.iterrows():
                positive_pairs.add((row["lncrna"], row["disease"]))

        all_lncrnas = list(self.node_indices["lncrna"].keys())
        all_diseases = list(self.node_indices["disease"].keys())

        if not positive_pairs or not all_lncrnas or not all_diseases:
            logger.warning("Insufficient data for negative sampling.")
            return []

        return self.negative_sampler.sample(
            positive_pairs, all_lncrnas, all_diseases
        )

    def _build_pyg_heterodata(
        self,
        node_features: Dict[str, torch.Tensor],
        negative_pairs: List[Tuple[str, str]],
    ) -> HeteroData:
        
        data = HeteroData()

        for ntype, feat in node_features.items():
            data[ntype].x = feat
            data[ntype].num_nodes = feat.shape[0]

        def edges_to_coo(
            df: pd.DataFrame, src_col: str, dst_col: str,
            src_type: str, dst_type: str, weight_col: str = "confidence"
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            src_indices = self.node_indices[src_type]
            dst_indices = self.node_indices[dst_type]

            src_list, dst_list, weights = [], [], []
            for _, row in df.iterrows():
                s = row[src_col]
                d = row[dst_col]
                if s in src_indices and d in dst_indices:
                    src_list.append(src_indices[s])
                    dst_list.append(dst_indices[d])
                    w = row.get(weight_col, 1.0)
                    weights.append(float(w) if pd.notna(w) else 1.0)

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_weight = torch.tensor(weights, dtype=torch.float)
            return edge_index, edge_weight

        if len(self.lncrna_disease_df) > 0:
            ei, ew = edges_to_coo(
                self.lncrna_disease_df, "lncrna", "disease",
                "lncrna", "disease", "confidence"
            )
            data["lncrna", "associated_with", "disease"].edge_index = ei
            data["lncrna", "associated_with", "disease"].edge_attr = ew
            data["disease", "rev_associated_with", "lncrna"].edge_index = ei.flip(0)
            data["disease", "rev_associated_with", "lncrna"].edge_attr = ew

        if len(self.mirna_disease_df) > 0:
            ei, ew = edges_to_coo(
                self.mirna_disease_df, "mirna", "disease",
                "mirna", "disease", "confidence"
            )
            data["mirna", "associated_with", "disease"].edge_index = ei
            data["mirna", "associated_with", "disease"].edge_attr = ew
            data["disease", "rev_associated_with", "mirna"].edge_index = ei.flip(0)
            data["disease", "rev_associated_with", "mirna"].edge_attr = ew

        if len(self.lncrna_mirna_df) > 0:
            ei, ew = edges_to_coo(
                self.lncrna_mirna_df, "lncrna", "mirna",
                "lncrna", "mirna", "confidence"
            )
            data["lncrna", "interacts_with", "mirna"].edge_index = ei
            data["lncrna", "interacts_with", "mirna"].edge_attr = ew
            data["mirna", "rev_interacts_with", "lncrna"].edge_index = ei.flip(0)
            data["mirna", "rev_interacts_with", "lncrna"].edge_attr = ew

        if len(self.ppi_df) > 0 and "gene_a" in self.ppi_df.columns:
            ei, ew = edges_to_coo(
                self.ppi_df, "gene_a", "gene_b",
                "mrna", "mrna", "combined_score"
            )
            if ew.numel() > 0:
                ew = ew / 1000.0
            data["mrna", "interacts_with", "mrna"].edge_index = ei
            data["mrna", "interacts_with", "mrna"].edge_attr = ew

        if len(self.disease_sim_df) > 0:
            ei, ew = edges_to_coo(
                self.disease_sim_df, "disease_a", "disease_b",
                "disease", "disease", "similarity"
            )
            data["disease", "similar_to", "disease"].edge_index = ei
            data["disease", "similar_to", "disease"].edge_attr = ew

        pos_src, pos_dst = [], []
        if len(self.lncrna_disease_df) > 0:
            for _, row in self.lncrna_disease_df.iterrows():
                l = row["lncrna"]
                d = row["disease"]
                if l in self.node_indices["lncrna"] and d in self.node_indices["disease"]:
                    pos_src.append(self.node_indices["lncrna"][l])
                    pos_dst.append(self.node_indices["disease"][d])

        neg_src, neg_dst = [], []
        for l, d in negative_pairs:
            if l in self.node_indices["lncrna"] and d in self.node_indices["disease"]:
                neg_src.append(self.node_indices["lncrna"][l])
                neg_dst.append(self.node_indices["disease"][d])

        data["lncrna", "associated_with", "disease"].pos_edge_label_index = (
            torch.tensor([pos_src, pos_dst], dtype=torch.long)
        )
        data["lncrna", "associated_with", "disease"].neg_edge_label_index = (
            torch.tensor([neg_src, neg_dst], dtype=torch.long)
        )

        data.validate(raise_on_error=True)

        return data

    def _print_summary(self):
        
        logger.info("\n" + "=" * 70)
        logger.info("GRAPH CONSTRUCTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nNode counts:")
        for ntype, indices in self.node_indices.items():
            logger.info(f"  {ntype:>10s}: {len(indices):>8,d} nodes")

        logger.info(f"\nEdge counts:")
        if self.hetero_data:
            for edge_type in self.hetero_data.edge_types:
                store = self.hetero_data[edge_type]
                if hasattr(store, "edge_index") and store.edge_index is not None:
                    ei = store.edge_index
                    logger.info(f"  {str(edge_type):>60s}: {ei.shape[1]:>8,d} edges")

        if len(self.lncrna_disease_df) > 0:
            melanoma_assoc = self.lncrna_disease_df[
                self.lncrna_disease_df["disease"].str.contains(
                    Config.TARGET_DISEASE, case=False
                )
            ]
            logger.info(
                f"\n  Melanoma-specific lncRNA associations: {len(melanoma_assoc)}"
            )

        logger.info(f"\n  Output: PyG HeteroData object ready for training")
        logger.info("=" * 70)

    def save(self, filename: str = "skcm_hetero_graph.pt"):
        
        if self.hetero_data is None:
            raise ValueError("Graph not built yet. Call build() first.")
        output_path = Config.PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.hetero_data, output_path)
        logger.info(f"Saved HeteroData to: {output_path}")
        return output_path

    def export_networkx(self, filename: str = "skcm_hetero_graph.graphml"):
        
        if self.G is None:
            raise ValueError("Graph not built yet. Call build() first.")
        output_path = Config.PROCESSED_DIR / filename
        nx.write_graphml(self.G, output_path)
        logger.info(f"Exported NetworkX graph to: {output_path}")
        return output_path


def main():
    x = HeteroGraphBuilder()
    data_out = x.build()


    x.save("skcm_hetero_graph.pt")
    x.export_networkx("skcm_hetero_graph.graphml")

    print("\n--- PyG HeteroData Summary ---")
    print(data_out)
    print(f"\nMetadata: {data_out.metadata()}")

    return data_out


if __name__ == "__main__":
    main()
