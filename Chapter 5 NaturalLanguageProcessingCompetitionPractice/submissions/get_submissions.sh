cd entity-extraction 
python predict_ee.py
cd ../entity-coreference
python main.py config.yaml
cd ../slot-filling
python predict_ef.py
cd ../entity-slot-alignment
python main.py config.yaml
cd .. 
python get_submissions.py

