import math
import dicom_global_params
import dicom_cat_params

def print_bit_arrangement_nd(Y: list, in_depth, in_width) -> list:
    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    for k in range(depth_Y):
        str_print = "assign dest_seq[" + str(depth_Y-1-k) + "] = {"
        for j in range(width_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                str_print = str_print + "1\'b0, "
            elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
                str_print = str_print + "1\'b1, "
            elif (j == width_Y-1):
                # temp_A1.append(str(matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]))
                str_print = str_print + "source_seq[" + str(in_depth-1-(Y[k][0][j]-1)) + \
                    "][" + str(in_width-1-(Y[k][1][j]-1)) + "]};"
            else:
                str_print = str_print + "source_seq[" + str(in_depth-1-(Y[k][0][j]-1)) + \
                    "][" + str(in_width-1-(Y[k][1][j]-1)) + "], "
        print(str_print + "\n")



def print_bit_arrangement_1d_to_nd(Y: list, in_width) -> list:
    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    for k in range(depth_Y):
        str_print = "assign dest_seq[" + str(depth_Y-1-k) + "] = {"
        for j in range(width_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                str_print = str_print + "1\'b0, "
            elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
                str_print = str_print + "1\'b1, "
            elif (j == width_Y-1):
                # temp_A1.append(str(matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]))
                str_print = str_print + "source_seq[" + str(in_width-1-(Y[k][1][j]-1)) + "]};"
            else:
                str_print = str_print + "source_seq[" + str(in_width-1-(Y[k][1][j]-1)) + "], "
        
        print(str_print + "\n")



def print_bit_arrangement_for_xy_generator(Y: list, in_depth, in_width) -> list:
    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    for k in range(depth_Y):
        if k == 0:
            print("if (IMAGE_K == 0) begin\n")
            str_print = "  assign dest_seq = {"
            for j in range(width_Y):
                if (j == width_Y-1):
                    # temp_A1.append(str(matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]))
                    str_print = str_print + "source_seq[" + str(in_depth-1-(Y[k][0][j]-1)) + "][" + str(in_width-1-(Y[k][1][j]-1)) + "]};\nend"
                else:
                    str_print = str_print + "source_seq[" + str(in_depth-1-(Y[k][0][j]-1)) + "][" + str(in_width-1-(Y[k][1][j]-1)) + "], "
        else:
            print("else if (IMAGE_K == " + str(k) + ") begin\n")
            str_print = "  assign dest_seq = {"
            for j in range(width_Y):
                if (j == width_Y-1):
                    str_print = str_print + "source_seq[" + str(in_depth-1-(Y[k][0][j]-1)) + "][" + str(in_width-1-(Y[k][1][j]-1)) + "]};\nend"
                else:
                    str_print = str_print + "source_seq[" + str(in_depth-1-(Y[k][0][j]-1)) + "][" + str(in_width-1-(Y[k][1][j]-1)) + "], "
        
        print(str_print + "\n")




# Y2_FAST_Cat
# print_bit_arrangement_nd(dicom_cat_params.Y2_FAST_Cat, 2, dicom_cat_params.m1_cat)

# Y4_FAST_Cat
# print_bit_arrangement_nd(dicom_cat_params.Y4_FAST_Cat, 2, dicom_cat_params.m1_cat)

# Yd_phi_source_Cat
# print_bit_arrangement_nd(dicom_cat_params.Yd_phi_source_Cat, 2, dicom_cat_params.m1_cat)

# Yd_phi_dest_Cat
print_bit_arrangement_nd(dicom_cat_params.Yd_phi_dest_Cat, 2, dicom_cat_params.m1_cat)




# Y1_FAST_Cat
# print_bit_arrangement_1d_to_nd(dicom_cat_params.Y1_FAST_Cat, dicom_cat_params.k1_cat)

# Y3_FAST_Cat
# print_bit_arrangement_1d_to_nd(dicom_cat_params.Y3_FAST_Cat, dicom_cat_params.k1_cat)



# print_bit_arrangement_for_xy_generator(dicom_cat_params.Yp_512x512_Cat, 2, dicom_cat_params.m1_cat)