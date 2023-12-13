import dicom_global_params as gp
import dicom_bit_functions as bf
from threading import Thread


# mt = multithread
# Use bit_pre_processing to call this
def bit_pre_processing_mt(
    XY, XY_new, phi_source, phi_dest, list_im_size, list_num_bits_pre
):
    XY_P_mt = []
    phi_source_P_mt = []
    phi_dest_P_mt = []

    threads = []

    def foo(
        XY_k,
        XY_new_k,
        phi_source_k,
        phi_dest_k,
        M_k,
        N_k,
        num_bits_pre_k,
        XY_P_mt,
        phi_source_P_mt,
        phi_dest_P_mt,
    ):
        XY_Pk, phi_source_Pk, phi_dest_Pk = bf.bit_pre_processing_k(
            XY_k, XY_new_k, phi_source_k, phi_dest_k, M_k, N_k, num_bits_pre_k
        )
        XY_P_mt.append(XY_Pk)
        phi_source_P_mt.append(phi_source_Pk)
        phi_dest_P_mt.append(phi_dest_Pk)
        return

    for k in range(gp.K):
        t = Thread(
            target=foo,
            kwargs=dict(
                XY_k=[XY[k][0], XY[k][1]],
                XY_new_k=[XY_new[k][0], XY_new[k][1]],
                phi_source_k=phi_source[k],
                phi_dest_k=phi_dest[k],
                M_k=list_im_size[k][0],
                N_k=list_im_size[k][1],
                num_bits_pre_k=list_num_bits_pre[k],
                XY_P_mt=XY_P_mt,
                phi_source_P_mt=phi_source_P_mt,
                phi_dest_P_mt=phi_dest_P_mt,
            ),
        )
        t.start()
        threads.append(t)

    # print("//////////////////////\n")
    # print("XY_P_mt: ", XY_P_mt)
    # print("phi_source_P_mt: ", phi_source_P_mt)
    # print("phi_dest_P_mt: ", phi_dest_P_mt)
    # print(XY)
    # print("//////////////////////\n")

    for t in threads:
        t.join()

    return XY_P_mt, phi_source_P_mt, phi_dest_P_mt
