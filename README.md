# This is the PASTormer V1 model

## Getting Started

To use your own data with this project, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.

    ```bash
    git clone https://github.com/MohsenAmiri79/PASTormer.git
    ```

2. **Navigate to the "data" directory**: Use the command line to navigate to the "data" directory within the cloned repository.

    ```bash
    cd your-repo/data
    ```

3. **Organize Your Data**:

   - **Training Data**:
     - Place your training images in the "training/x" directory.
     - Place the corresponding ground truth images in the "training/y" directory.

   - **Validation Data**:
     - Place your validation/query images in the "validation/x" directory.
     - Place the corresponding ground truth images in the "validation/y" directory.

4. **Commit Your Changes**: Once you've organized your data, commit your changes to the repository.

    ```bash
    git add .
    git commit -m "Add my data files"
    git push
    ```

5. **Use the Project**: You can now use the project with your own data by following the project's instructions.

## Project Dependencies

einops==0.6.1
matplotlib==3.7.3
numpy==1.25.2
Pillow==10.0.0
pytorch-msssim==1.0.0
torch==2.0.1
torchvision==0.15.2

## License

This project is licensed under the MIT License. See the [LICENSE FILE](LICENSE.md) for details.
