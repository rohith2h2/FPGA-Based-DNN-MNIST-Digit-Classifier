
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdlib.h>

#include "hwlib.h"
#include "socal/socal.h"
#include "socal/hps.h"
#include "hps_0.h"

#include "layer1_bias.h"
#include "layer1_weight.h"
#include "layer2_bias.h"
#include "layer2_weight.h"
#include "input_0.h"
#include "input_1.h"
#include "input_2.h"
#include "input_3.h"
#include "input_4.h"
#include "input_5.h"
#include "input_6.h"
#include "input_7.h"
#include "input_8.h"
#include "input_9.h"

#define HW_REGS_BASE (ALT_STM_OFST)
#define HW_REGS_SPAN (0x04000000)
#define HW_REGS_MASK (HW_REGS_SPAN - 1)

volatile unsigned long *h2p_lw_hex0_addr = NULL;
volatile unsigned long *h2p_lw_hex1_addr = NULL;
volatile unsigned long *h2p_lw_hex2_addr = NULL;
volatile unsigned long *h2p_lw_hex3_addr = NULL;
volatile unsigned long *h2p_lw_hex4_addr = NULL;
volatile unsigned long *h2p_lw_hex5_addr = NULL;
static unsigned char szMap[] = {64, 121, 36, 48, 25, 18, 2, 120, 0, 16, 127};

static volatile unsigned long long *dout = NULL;
static float *img_virtual_base = NULL;
static float *b_virtual_base = NULL;
static float *w_virtual_base = NULL;

int full_init(int *virtual_base) {
    int fd;
    void *virtual_space;

    if ((fd = open("/dev/mem", (O_RDWR | O_SYNC))) == -1) {
        printf("can't open the file");
        return fd;
    }

    virtual_space = mmap(NULL, HW_REGS_SPAN, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, HW_REGS_BASE);

    dout = virtual_space + ((unsigned)(ALT_LWFPGASLVS_OFST + PREDIT_0_MY_PREDIT_INTERNAL_INST_AVS_CRA_BASE) & (unsigned)(HW_REGS_MASK));
    b_virtual_base = virtual_space + ((unsigned)(ALT_LWFPGASLVS_OFST + PREDIT_0_MY_PREDIT_INTERNAL_INST_AVS_B_BASE) & (unsigned)(HW_REGS_MASK));
    w_virtual_base = virtual_space + ((unsigned)(ALT_LWFPGASLVS_OFST + PREDIT_0_MY_PREDIT_INTERNAL_INST_AVS_W_BASE) & (unsigned)(HW_REGS_MASK));
    img_virtual_base = virtual_space + ((unsigned)(ALT_LWFPGASLVS_OFST + PREDIT_0_MY_PREDIT_INTERNAL_INST_AVS_IMG_BASE) & (unsigned)(HW_REGS_MASK));
    *virtual_base = virtual_space;
    return fd;
}

int main() {
    int fd, virtual_base, i;
    fd = full_init(&virtual_base);
    float *image[10] = {input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9};

    memcpy(w1_virtual_base, layer1_weight, 784*sizeof(float));
    memcpy(b2_virtual_base, layer2_bias, 784*10 * sizeof(float));

    for (i = 0; i < 10; i++) {
        memcpy(img_virtual_base, image[i], 784 * sizeof(float));
        while ((*dout & 1) != 0);
        *(dout + 2) = 1;
        *(dout + 3) = 1;
        *(dout + 1) = 1;

        while ((*(dout + 3) & 0x2) == 0);
        printf("input:%d result:%d\n", i, *(dout + 4));
        *(dout + 1) = 0;
    }

    *h2p_lw_hex0_addr = szMap[*(dout + 4) % 10];

    if (munmap(virtual_base, HW_REGS_SPAN) == -1) {
        printf("Unmap failed..\n");
        close(fd);
    }

    close(fd);
    return 0;
}
