source .venv/bin/activate

# python streamline/ExploratoryAnalysisMain.py --data-path /home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/data_input --out-path /home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/results --exp-name myoxia_streamline --inst-label id --class-label conclusion --run-parallel False --fi /home/meyer/code-project/STREAMLINE/EHRoes/ignore_feature.csv
# python streamline/CustomPreProcessing.py /home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/data_input
# python streamline/DataPreprocessingMain.py --out-path /home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/results --exp-name myoxia_streamline --run-parallel False --scale False
# python streamline/FeatureImportanceMain.py --out-path /home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/results --exp-name myoxia_streamline --run-parallel False
# python streamline/FeatureSelectionMain.py --out-path /home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/results --exp-name myoxia_streamline --run-parallel False --top-features 10
python streamline/ModelMain.py --out-path /home/meyer/code-project/AutoML-Pipe/AutoML-Pipe/results --exp-name myoxia_streamline --run-parallel False --do-NB True --do-LR True --do-DT True --do-RF True --do-GB True --do-XGB True --do-LGB True --do-CGB True --do-SVM True --do-ANN True --do-eLCS True --do-XCS True --do-ExSTraCS True --n-trials 100 --timeout 600 --iter 10000 --N 200
# python streamline/StatsMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False
# python streamline/DataCompareMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False
# python streamline/PDF_ReportMain.py --out-path /myoutputpath --exp-name hcc_demo --run-parallel False
# python streamline/ApplyModelMain.py --out-path /myoutputpath --exp-name hcc_demo --rep-data-path /myrepdatapath/STREAMLINE/DemoRepData  --dataset /mydatapath/STREAMLINE/DemoData/hcc-data_example.csv --run-parallel False
# python streamline/PDF_ReportMain.py --training False --out-path /myoutputpath --exp-name hcc_demo --rep-data-path /myrepdatapath/STREAMLINE/DemoRepData  --dataset /mydatapath/STREAMLINE/DemoData/hcc-data_example.csv --run-parallel False
# python streamline/FileCleanup.py --out-path /myoutputpath --exp-name hcc_demo