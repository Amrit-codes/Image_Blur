from mpi4py import MPI
import numpy as np
from PIL import Image, ImageFilter

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to blur an image section
def blur_image_section(image_section):
    return image_section.filter(ImageFilter.BLUR)

# Check if it's the master process
if rank == 0:
    # Load the image
    input_image = Image.open("input_image.jpg")
    width, height = input_image.size

    # Divide the image into sections based on the number of processes
    section_height = height // size
    remainder = height % size

    # Distribute sections to worker processes
    for i in range(1, size):
        start_row = i * section_height
        end_row = (i + 1) * section_height
        if i == size - 1:
            end_row += remainder
        image_section = input_image.crop((0, start_row, width, end_row))
        comm.send(image_section, dest=i)

    # Process the master's section
    master_section = input_image.crop((0, 0, width, section_height))
    blurred_section = blur_image_section(master_section)

    # Receive and assemble sections from worker processes
    for i in range(1, size):
        received_section = comm.recv(source=i)
        start_row = i * section_height
        end_row = (i + 1) * section_height
        if i == size - 1:
            end_row += remainder
        input_image.paste(received_section, (0, start_row))

    # Save the blurred image
    input_image.save("blurred_image.jpg")

# Worker processes receive and process their sections
else:
    image_section = comm.recv(source=0)
    blurred_section = blur_image_section(image_section)
    comm.send(blurred_section, dest=0)
