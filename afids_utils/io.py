"""General methods for loading and saving files associated with AFIDs"""
from __future__ import annotations

from os import PathLike

from afids_utils.afids import AfidSet
from afids_utils.extensions.fcsv import load_fcsv

def load(afids_fpath: PathLike[str] | str) -> AfidSet:
    """
    Load an AFIDs file

    Parameters
    ----------
    afids_fpath : os.PathLike[str] | str
        Path to .fcsv or .json file containing AFIDs information

    Returns
    -------
    AfidSet
        Set of anatomical fiducials containing spatial coordinates and metadata

    Raises
    ------
    IOError
        If extension to fiducial file is not .fcsv or .json or does not exist
    """
    afids_fpath = Path(afids_fpath)

    # Check if file exists
    if not afids_fpath.exists():
        raise IOError("Provided AFID file does not exist")

    afids_fpath_ext = afids_fpath.suffix

    # Loading fcsv
    if afids_fpath_ext = ".fcsv":
        load_fcsv(afids_fpath)
    # Loading json
    # if afids_fpath_ext = ".json":
    #   load_json(afids_path)
    else:
        raise IOError("Invalid file extension")



# def afids_to_fcsv(
#     afid_coords: NDArray[np.single],
#     fcsv_output: PathLike[str] | str,
# ) -> None:
#     """
#     Save AFIDS to Slicer-compatible .fcsv file

#     Parameters
#     ----------
#     afid_coords : numpy.ndarray[shape=(N, 3), dtype=numpy.single]
#         Floating-point NumPy array containing spatial coordinates (x, y, z) of
#         `N` AFIDs

#     fcsv_output : os.PathLike[str] | str
#         Path of file (including filename and extension) to save AFIDs to

#     """
#     # Read in fcsv template
#     with resources.open_text(
#         "afids_utils.resources", "template.fcsv"
#     ) as template_fcsv_file:
#         header = [template_fcsv_file.readline() for _ in range(3)]
#         reader = csv.DictReader(
#             template_fcsv_file, fieldnames=list(FCSV_FIELDNAMES.keys())
#         )
#         fcsv = list(reader)

#     # Check to make sure shape of AFIDs array matches expected template
#     if afid_coords.shape[0] != len(fcsv):
#         raise TypeError(
#             f"Expected {len(fcsv)} AFIDs, but received {afid_coords.shape[0]}."
#         )
#     if afid_coords.shape[1] != 3:
#         raise TypeError(
#             "Expected 3 spatial dimensions (x, y, z),"
#             f"but received {afid_coords.shape[1]}."
#         )

#     # Loop over fiducials and update with fiducial spatial coordinates
#     for idx, row in enumerate(fcsv):
#         row["x"] = afid_coords[idx][0]
#         row["y"] = afid_coords[idx][1]
#         row["z"] = afid_coords[idx][2]

#     # Write output fcsv
#     with open(fcsv_output, "w", encoding="utf-8", newline="") as out_fcsv_file:
#         for line in header:
#             out_fcsv_file.write(line)
#         writer = csv.DictWriter(
#             out_fcsv_file, fieldnames=list(FCSV_FIELDNAMES.keys())
#         )
#         for row in fcsv:
#             writer.writerow(row)
