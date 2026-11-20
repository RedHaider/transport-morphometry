from transport_morph.experiment import dataset,evaluate
def test_shape():
 x,y=dataset(20,2);assert x.shape==(20,120) and len(y)==20
def test_metrics():assert all(0<=v["auroc"]<=1 for v in evaluate(3).values())

