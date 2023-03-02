# `Shell-ocp` based on Open Catalyst Project

[![CircleCI](https://circleci.com/gh/Open-Catalyst-Project/ocp.svg?style=shield)](https://circleci.com/gh/Open-Catalyst-Project/ocp)
[![codecov](https://codecov.io/gh/Open-Catalyst-Project/ocp/branch/codecov/graph/badge.svg?token=M606LH5LK6)](https://codecov.io/gh/Open-Catalyst-Project/ocp)

## Project description

- Put the file in the correct dictionary
- `.pt` the pre-trained parameters is in the release.
- Put the file in the correct dictionary
- `.lmdb` `.lmdb-lock` the dataset is in the dgx /shareddata/open-catalyst/slab/datasetss/slabs_rare.lmdb    /shareddata/open-catalyst/slab/datasetss/slabs_rare.lmdb-lock 

- The slab generation code task is under the `Shell_repo/ocp/slab_generation/`


## Installation-step 1

See [installation instructions](https://github.com/Open-Catalyst-Project/ocp/blob/main/INSTALL.md).

<!-- ## Installation-step 2

* Install specific versions of Pymatgen and ASE: `pip install pymatgen==2020.4.2 `
* Install Catkit from Github: `pip install git+https://github.com/ulissigroup/CatKit.git catkit`
* Clone this repo and install with: `pip install -e ..` -->



## Acknowledgements

* This work is an extension work based on OCP dataset, machine learning model and most of the scripts.

`ocp` is the [Open Catalyst Project](https://opencatalystproject.org/)'s
library of state-of-the-art machine learning algorithms for catalysis.

<!-- <div align="left">
    <img src="https://user-images.githubusercontent.com/1156489/170388229-642c6619-dece-4c88-85ef-b46f4d5f1031.gif">
</div> -->

It provides training and evaluation code for tasks and models that take arbitrary
chemical structures as input to predict energies / forces / positions, and can
be used as a base scaffold for research projects. For an overview of tasks, data, and metrics, please read our papers:
 - [OC20](https://arxiv.org/abs/2010.09990)
 - [OC22](https://arxiv.org/abs/2206.08917)


## License

`ocp` is released under the [MIT](https://github.com/Open-Catalyst-Project/ocp/blob/main/LICENSE.md) license.

## Citing `ocp`

If you use this codebase in your work, please consider citing:

```bibtex
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```
