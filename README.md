The **Roman_config.py** includes all the parameters and file paths for running the pipeline. You can adjust them there.

The **build_data_and_random.py** is used for generate data and random catalogs for the galaxies and interlopers. Currently, only S3 is included, the percentage can be changed in the Roman_config.py.

The **calculate_2pcf.py** is used for calculating the two point correlation function.

The **run_rascalC.py** is used for generating RascalC covariance matrix. Since the post_processing has already been done in this script, we elimiated the make_cov in the pipeline.

The **fit_bao.py** is used for BAO fitting.

The **run_pipeline.py** is to run the full pipeline. Since all of the functions can be called independently, you can remove steps in the run_pipeline.py.

The **Test_run.ipynb** is the jupyter notebook for running test.
