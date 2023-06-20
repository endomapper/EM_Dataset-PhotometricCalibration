# EM Dataset: Photometric calibration
**Authors:** [Víctor M. Batlle](http://webdiis.unizar.es/~vmbatlle/), [José M. M. Montiel](http://webdiis.unizar.es/~josemari/), [Juan D. Tardos](http://webdiis.unizar.es/~jdtardos/).


This is a software to obtain an accurate calibration of the endoscope's **photometry**. Its main functionality is the photometric calibration of the endoscope's camera and light, using the EndoMapper calibration sequences.

### Related Publications:

[Photometric22] Víctor M. Batlle, J. M. M. Montiel and Juan D. Tardós, [**"Photometric single-view dense 3D reconstruction in endoscopy"**](https://ieeexplore.ieee.org/abstract/document/9981742), *2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, Kyoto, Japan, 2022, pp. 4904-4910. [PDF](https://arxiv.org/pdf/2204.09083.pdf)
```
@inproceedings{batlle2022photometric,
  title={Photometric single-view dense 3D reconstruction in endoscopy},
  author={Batlle, V{\'\i}ctor M and Montiel, Jos{\'e} MM and Tard{\'o}s, Juan D},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4904--4910},
  year={2022},
  organization={IEEE}
}
```

# 2. Prerequisites
We have tested the software in **Ubuntu 20.04**, but it should be executable on other platforms.

## FFmpeg
We use [FFmpeg](https://ffmpeg.org/) to extract the individual frames from calibration sequences.

```sh
sudo apt install ffmpeg
```

## Python 
We use [Python](https://www.python.org) for endoscope calibration and depth estimation. **Required 3.X**. **Recommended 3.8.10**

```sh
sudo apt install python
```

### Required packages:

* Numpy 1.17.4 (w\ nptyping 1.4.4)
* OpenCV 3.4.17
* Scipy 1.8.0
* Matplotlib 3.5.1
* Tqdm 4.61.2

You can install these Python packages with:

```sh
pip3 install -r requirements.txt 
```

## Vicalib

We use [Vicalib](https://github.com/arpg/vicalib) for calibrating the camera intrinsic parameters and obtaining camera poses during photometric calibration.

**Required** to compile and install the slightly [*modified version*](https://github.com/vmbatlle/vicalib) of Vicalib.


# 2. EndoMapper photometric calibration

We provide a script to calibrate all endoscopes in the EndoMapper dataset.

Choose one method:

- Auto: execute `./run` and follow the instructions on the terminal.
- Manual: follow steps below.

## Manual calibration

1. Download the [EndoMapper dataset](https://doi.org/10.7303/syn26707219) [[Azagra et al., 2022](https://arxiv.org/abs/2204.14240.pdf)].
   The calibration sequences should be accesible at `path/to/dataset/Calibrations/Endoscope_XX`.
2. Check that the folder structure is as follows:
  ```bash
  $ ls -R path/to/dataset

  ./path/to/dataset:
  Calibrations

  ./path/to/dataset/Calibrations:
  Endoscope_01  Endoscope_02  Endoscope_03  Endoscope_04 ...

  ./path/to/dataset/Calibrations/Endoscope_01:
  Endoscope_01.mov  Endoscope_02_geometrical.xml
  ...
  ```
3. Run the following script to copy required files:
```sh
./utils/prepare_dataset.sh /path/to/dataset
```
4. Run the following script to calibrate all endoscopes:
```sh
./utils/calibrate_dataset.sh /path/to/dataset
```

## Calibration results

For each Endoscope_XX the calibration generates:

- `Endoscope_XX_geometrical.xml` with the geometric parameters.
- `Endoscope_XX_photometrical.xml` with the photometric parameters.

# 3. Calibrate your own hardware

1. Record a calibration sequence with the [Vicalib pattern](misc/big_pattern.pdf). Perform a camera motion similar to the sequences in the EndoMapper dataset.
1. Export the recording to a `.MOV` file inside `path/to/dataset/Calibrations/Endoscope_XX`.
1. Execute `./run`. Note that for non-fisheye cameras you can use `poly2` camera model.
