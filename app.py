import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

st.set_page_config(page_title="Side Effect Classifier", layout="centered")

st.title("⚠️ Drug Side Effect Classifier")
st.write("Predicts common side effect risks from chemical structure")

drug_dict = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "Omeprazole": "COc1ccc2[nH]c(nc2c1)S(=O)Cc3ncc(C)c(OC)c3C"
}

selected = st.selectbox("Or select a common drug:", list(drug_dict.keys()))
smiles = st.text_input("Enter SMILES:", drug_dict[selected])

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES")
    else:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)

        # Simple risk models (for learning/demo)
        gi_risk = min(95, max(5, (logp * 15) + (50 if mw > 350 else 10) + 10))
        cns_risk = min(95, max(5, (logp * 18) - (tpsa * 0.4) + 20))
        cardiac_risk = min(95, max(5, (logp * 10) + (hbd * 5) + 15))
        liver_risk = min(95, max(5, (mw / 5) + (logp * 8)))

        st.subheader("Predicted Side Effect Risk")

        col1, col2 = st.columns(2)
        col1.metric("GI Risk (Nausea, Ulcer)", f"{gi_risk:.0f}%")
        col2.metric("CNS Risk (Dizziness, Headache)", f"{cns_risk:.0f}%")
        col1.metric("Cardiac Risk", f"{cardiac_risk:.0f}%")
        col2.metric("Liver Risk", f"{liver_risk:.0f}%")

        df = pd.DataFrame({
            "Side Effect": ["GI", "CNS", "Cardiac", "Liver"],
            "Risk %": [gi_risk, cns_risk, cardiac_risk, liver_risk]
        })
        st.bar_chart(df.set_index("Side Effect"))

        if max(gi_risk, cns_risk, cardiac_risk, liver_risk) > 70:
            st.warning("High risk predicted. Structural modification advised.")
        else:
            st.success("Low to moderate risk profile")

st.caption("Demo model for education. Not for clinical use.")
