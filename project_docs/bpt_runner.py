
import sys
sys.path.append(r"D:\ClothesNetData\project_docs")

from bpt_converter import batch_convert_to_bpt

# Run the BPT conversion with specified parameters
batch_convert_to_bpt(
    input_dir=r"D:\ClothesNetData_Processed",
    output_dir=r"D:\ClothesNetData_QuadBPT_Converted",
    block_size=16,
    patch_size=4
)
