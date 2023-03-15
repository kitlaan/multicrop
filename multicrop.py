#!/usr/bin/env python3

from typing import Any, List, Optional
import argparse
import cv2
import imutils
import imutils.contours
from imutils.perspective import four_point_transform
import numpy
import os
import sys

referenceImage: Optional[cv2.Mat] = None


def debugRender(image: cv2.Mat, filename = None):
    render = imutils.resize(image, height=1000)
    if render.dtype.itemsize == 2:
        render = (render >> 8).astype("uint8")

    if filename:
        cv2.imwrite(filename, render)
    else:
        cv2.imshow("image", render)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def addBorder(image: cv2.Mat) -> cv2.Mat:
    BORDER_SIZE = 100

    if image.dtype.itemsize == 2:
        white = [65535, 65535, 65535]
    elif image.dtype.itemsize == 1:
        white = [255, 255, 255]
    else:
        # FIXME
        sys.exit(f"unexpected bit depth {image.dtype.itemsize}")

    return cv2.copyMakeBorder(
        image,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=white,
    )


def setReferenceImage(filename: str) -> bool:
    global referenceImage

    image = cv2.imread(filename)
    if image is None:
        return False

    # add a fixed-size border
    image = addBorder(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    referenceImage = cv2.GaussianBlur(image, (9, 9), 0)

    return True


def findContours(image: cv2.Mat) -> List[Any]:
    # convert image to 8-bit to do operations
    if image.dtype.itemsize == 2:
        image = (image >> 8).astype("uint8")
    elif image.dtype.itemsize == 1:
        pass
    else:
        # FIXME
        sys.exit(f"unexpected bit depth {image.dtype.itemsize}")

    global referenceImage
    if referenceImage is not None and image.shape[:2] == referenceImage.shape[:2]:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.GaussianBlur(grayImage, (9, 9), 0)

        fgmask = cv2.subtract(referenceImage, grayImage)

        # anything not near-black is treated as "image"
        # 24 worked because it worked for "1990 - 018"
        _, fgmask = cv2.threshold(fgmask, 24, 255, cv2.THRESH_BINARY)
    else:
        fgmask = image.copy()
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # 240 picked because it worked for "1991 - 046"
        _, fgmask = cv2.threshold(fgmask, 240, 255, cv2.THRESH_BINARY_INV)

    # the scanner bed holds at most 3x2 pictures on the scanner bed, so each contour
    # should have an area of at least width/3 x height/3
    minArea = (image.shape[0] * image.shape[1]) / 9

    # find contour and reduce to those meeting probable size.
    contours = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        x for x in imutils.grab_contours(contours) if cv2.contourArea(x) > minArea
    ]

    if len(contours) == 0:
        return []

    contours, _ = imutils.contours.sort_contours(contours)
    return contours


def detect_corners(contour: List[Any]):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_corners = cv2.approxPolyDP(contour, epsilon, True)

    # remove singleton dimensions
    approx_corners = numpy.squeeze(approx_corners)

    # figure out the minimum bounding for these points
    box = cv2.minAreaRect(approx_corners)
    box = cv2.boxPoints(box)
    return box

    # FIXME: alternatively (which doesn't work). I think this should really use hough_lines_intersection
    # to find the corner points

    # find the outermost 4 corners by finding the closest to the bounding box
    corners = []
    for corner in box:
        pts = sorted(approx_corners, key=lambda x: cv2.norm(x - corner))
        corners.append(pts[0])

    return numpy.array(corners)


def processFile(filename: str, output: str, options):
    basename, _ = os.path.splitext(os.path.basename(filename))
    print(filename)

    # add a fixed-size white border to allow contour detection
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    image = addBorder(image)

    # locate the bounding boxes
    contours = findContours(image)

    if options.debug_contour:
        debug = image.copy()
        debug = cv2.drawContours(debug, contours, -1, (65535, 0, 0), 20)

    print(f"{filename}: Found {len(contours)} images")

    for i, c in enumerate(contours):
        outfile = os.path.join(output, f"{basename}-{i}.png")

        if options.deskew == "corner":
            box = detect_corners(c)
        else:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)

        if options.debug_contour:
            debug = cv2.drawContours(debug, [numpy.intp(box)], 0, (0, 65535, 32767), 16)

        if options.no_output:
            continue

        cropped = four_point_transform(image, box)
        if options.crop:
            crop = options.crop
            cropped = cropped[crop:-crop, crop:-crop]
        cv2.imwrite(outfile, cropped)

    if options.debug_contour:
        debugfile = os.path.join(output, f"{basename}-debug.png")
        debugRender(debug, debugfile)


def main():
    parser = argparse.ArgumentParser(description="Crop and deskew images.")
    parser.add_argument("-i", "--input", help="Input path or filename")
    parser.add_argument("-o", "--output", default="output", help="Output path")
    parser.add_argument("-r", "--reference", help="Reference image filename")
    group = parser.add_argument_group()
    group.add_argument("--crop", type=int, default=0, help="Crop border")
    group.add_argument(
        "--deskew",
        help="Method of deskewing",
        choices=["rect", "corner"],
        default="rect",
    )
    group = parser.add_argument_group()
    group.add_argument("--debug-contour", action="store_true")
    group.add_argument("--no-output", action="store_true")
    args = parser.parse_args()

    # TODO: --aspect-hint=4x3 allow an option to tell the expected picture size aspect ratio,
    #       which would allow contour detection to "fill in" some of the more odd detections

    if args.reference:
        if not os.path.isfile(args.reference):
            sys.exit(f"Could not find reference image: {args.reference}")
        if not setReferenceImage(args.reference):
            sys.exit(f"Invalid reference image: {args.reference}")

    if not args.input:
        parser.print_help()
    elif os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            fullpath = os.path.join(args.input, filename)
            if cv2.haveImageReader(fullpath):
                processFile(fullpath, args.output, options=args)
    elif os.path.isfile(args.input):
        processFile(args.input, args.output, options=args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
