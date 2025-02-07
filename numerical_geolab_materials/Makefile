materials_path="/path_to_numerical_geolab/numerical_geolab/numerical_geolab_materials/UMATERIALS"

#mode="debug"
mode="release"

all:
	make -C $(materials_path)/"CAUCHY3D-ASYMCAMCLAY" $(mode)
	make -C $(materials_path)/"CAUCHY3D-AUSSOIS-PR-TEMP" $(mode)
	make -C $(materials_path)/"CAUCHY3D-DP" $(mode)
	make -C $(materials_path)/"CAUCHY3D-DP-PR-TEMP" $(mode)
	make -C $(materials_path)/"COSSERAT3D" $(mode)
	make -C $(materials_path)/"COSSERAT3D-THM" $(mode)