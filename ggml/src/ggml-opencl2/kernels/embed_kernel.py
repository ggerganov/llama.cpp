import sys

def main():
  if len(sys.argv) != 3:
    print("Usage: python embed_kernel.py <input_file> <output_file>")
    exit(1)

  ifile = open(sys.argv[1], "r")
  ofile = open(sys.argv[2], "w")

  ofile.write("R\"(\n\n")
  ofile.write(ifile.read())
  ofile.write("\n)\"")

  ifile.close()
  ofile.close()

if __name__ == "__main__":
  main()