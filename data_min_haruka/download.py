from mp_api.client import MPRester 
import pickle
with MPRester("api_key") as mpr:
    docs = mpr.materials.summary.search(
        elements=["Fe"],
        num_elements=(2, 2)
        )
    docs_dict = [doc.model_dump() for doc in docs]
    with open("fe_docs.pkl", "wb") as f:
        pickle.dump(docs_dict, f)

    example_doc = docs[0]
    mpid = example_doc.material_id
    formula = example_doc.formula_pretty
    print(formula)
    print("done")
