import cv2
import numpy as np
import os

# Load the fisheye image
image_path = "C:/Users/User/Downloads/FYP/split_images/Splitted/fisheye1.jpg"
fisheye_image = cv2.imread(image_path)

# Original (height, width, rgb), [:2] is slicing the first two values from this tuple,
height, width = fisheye_image.shape[:2] 

# Output directory
output_dir = r"C:/Users/User/Downloads/FYP/split_images/Undistorted"
os.makedirs(output_dir, exist_ok=True)

# Offsets and Ranges
# Offset is maximum
offset = {
    'fx': width, 'fy': width,  # fx, fy: allow from -width to +width
    'cx': width, 'cy': height,  # cx, cy: allow from -width to +width
    'k1': 1000, 'k2': 1000,     # k1, k2 scaled by 1000
    'zoom': 10,
}

# Range allow for minumum and maximum
range_ = {
    'fx': width * 2, 'fy': width * 2,
    'cx': width * 2, 'cy': height * 2,
    'k1': 2000, 'k2': 2000,
    'zoom': 30,
    'rotation': 7  # 0 to 7 => 0° to 315°
}

# Initial values
init_vals = {
    'fx': int(width / 1.7),
    'fy': int(width / 1.7),
    'cx': int(width / 2),
    'cy': int(height / 2),
    'k1': int(-0.34 * 1000),
    'k2': int(-0.21 * 1000),
    'zoom': 0,
    'rotation': 0
}

# For Reference
# K = np.array([[fx,  0,  cx],  # Focal length in x, Optical center x
#               [ 0, fy,  cy],  # Focal length in y, Optical center y
#               [ 0,  0,   1]], dtype=np.float32)

# Adjust initial values by offset for trackbar
init_trackbar = {k: v + offset.get(k, 0) for k, v in init_vals.items()}

# Create two separate windows, Controls and Undistorted
# Window for controls
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 400, 500)
# Window for real time changes
cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Undistorted", width, height)

# Create trackbars 
for key in ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'zoom']:
    # Create trackbars in the "Controls" window.
    # If the key is k1 or k2, we label it with (x1000) to indicate scaling.
    # This means k1 and k2 are actually float values (like -0.34), 
    # but we scale them up by 1000 to make the slider integer-based.
    cv2.createTrackbar(key if 'k' not in key else f"{key} (x1000)", "Controls", init_trackbar[key], range_[key], lambda x: None)
    # (Display label, Window name, Initial position (offset applied), Maximum range value, Dummy function)
cv2.createTrackbar("rotation", "Controls", init_vals['rotation'], range_['rotation'], lambda x: None)
cv2.createTrackbar("Default", "Controls", 0, 1, lambda x: None)

# Previous state holder to detect button press
prev_restore_val = 0

# Restore Defaults Function 
def restore_defaults():
    for key in ['fx', 'fy', 'cx', 'cy']:
        cv2.setTrackbarPos(key, "Controls", init_vals[key] + offset[key])
        cv2.setTrackbarPos("k1 (x1000)", "Controls", int(init_vals['k1'] + offset['k1']))
        cv2.setTrackbarPos("k2 (x1000)", "Controls", int(init_vals['k2'] + offset['k2']))
        cv2.setTrackbarPos("zoom", "Controls", int(init_vals['zoom']+ offset['zoom']))
        cv2.setTrackbarPos("rotation", "Controls", init_vals['rotation'])
        cv2.setTrackbarPos("Default", "Controls", 0)  # Reset button
    print("🔄 Parameters restored to default.")

# Helper functions
# This function reads the values from all the sliders and converts them into usable numbers.
# Since all sliders were offset positively to allow negative ranges, we subtract the offsets.
# For k1 and k2, we divide by 1000 to get back their original float representation.
def get_slider_values():
    fx = cv2.getTrackbarPos("fx", "Controls") - offset['fx']
    fy = cv2.getTrackbarPos("fy", "Controls") - offset['fy']
    cx = cv2.getTrackbarPos("cx", "Controls") - offset['cx']
    cy = cv2.getTrackbarPos("cy", "Controls") - offset['cy']
    k1 = (cv2.getTrackbarPos("k1 (x1000)", "Controls") - offset['k1']) / 1000.0
    k2 = (cv2.getTrackbarPos("k2 (x1000)", "Controls") - offset['k2']) / 1000.0
    zoom = cv2.getTrackbarPos("zoom", "Controls") / 10.0
    rotation = cv2.getTrackbarPos("rotation", "Controls") * 45
    return fx, fy, cx, cy, k1, k2, zoom, rotation

# For naming files based on parameters
def format_param(val):
    prefix = "P" if val >= 0 else "N"
    return f"{prefix}{abs(val):.2f}"

# Rotating Images
def rotate_image(img, angle):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

# Apply Changes to Images
def update_image():
    fx, fy, cx, cy, k1, k2, zoom, rotation = get_slider_values()

    # Original camera matrix (without zoom)
    K_in = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)

    # New camera matrix with zoom applied
    # Zoom is applied by scaling the focal lengths (fx and fy)
    # Because increasing the focal length zooms in (narrows FOV), 
    # and decreasing it zooms out (widens FOV).
    K_out = np.array([[fx * zoom, 0, cx],
                      [0, fy * zoom, cy],
                      [0,  0,  1]], dtype=np.float64)

    D = np.array([k1, k2, 0, 0], dtype=np.float64)

    # Undistortion using fisheye model
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_in, D, np.eye(3), K_out, (width, height), cv2.CV_16SC2)
    undistorted = cv2.remap(fisheye_image, map1, map2, interpolation=cv2.INTER_LINEAR)

    rotated = rotate_image(undistorted, rotation)
    return rotated, fx, fy, cx, cy, k1, k2, zoom, rotation

# Main loop 
while True:
    img, fx, fy, cx, cy, k1, k2, zoom, rotation = update_image()
    cv2.imshow("Undistorted", img)

    key = cv2.waitKey(1) & 0xFF

    # Detect 'Restore Default' button press 
    restore_val = cv2.getTrackbarPos("Default", "Controls")
    if restore_val == 1 and prev_restore_val == 0:
        restore_defaults()
    prev_restore_val = restore_val

    # Save image
    if key == ord('s'):
        fname = f"{format_param(fx)}_{format_param(fy)}_{format_param(cx)}_{format_param(cy)}_{format_param(k1)}_{format_param(k2)}_Z{zoom:.2f}_R{rotation}_img.jpg"
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, img)
        print(f"✅ Saved: {save_path}")

    # Allow manual input
    elif key == ord('m'):
        try:
            fx = float(input("fx: "))
            fy = float(input("fy: "))
            cx = float(input("cx: "))
            cy = float(input("cy: "))
            k1 = float(input("k1: "))
            k2 = float(input("k2: "))
            zoom = float(input("zoom: "))
            rotation = int(input("rotation (0-315): "))

            # Update sliders
            cv2.setTrackbarPos("fx", "Controls", int(fx + offset['fx']))
            cv2.setTrackbarPos("fy", "Controls", int(fy + offset['fy']))
            cv2.setTrackbarPos("cx", "Controls", int(cx + offset['cx']))
            cv2.setTrackbarPos("cy", "Controls", int(cy + offset['cy']))
            cv2.setTrackbarPos("k1 (x1000)", "Controls", int(k1 * 1000 + offset['k1']))
            cv2.setTrackbarPos("k2 (x1000)", "Controls", int(k2 * 1000 + offset['k2']))
            cv2.setTrackbarPos("zoom", "Controls", int(zoom * 10))
            cv2.setTrackbarPos("rotation", "Controls", rotation // 45)
        except:
            print("⚠️ Invalid manual input.")
    
    elif key == 27:
        break

cv2.destroyAllWindows()
