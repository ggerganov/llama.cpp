#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#include "ggml-cuda/ggml-cuda.cu"
//ToDo: This needs to be properly integrated

const int GGML_CUDA_CC_OFFSET_AMD = 0x1000000;

struct target {
    char id[256];
    char major;
    char minor;
    char step;
};

int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if (strstr(archName, "-generic")) {
            archName[archLen - 8] = '\0';
            char * pch;
            if (pch = strtok(archName, "-")) {
                archMajor = (int)strtoul(pch, 0, 16);
                if (pch = strtok(NULL, "-")) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;

    // be inclusive of the full gfx8 line for backward compatibility (Carrizu APUs, etc.)
    if (archMajor != 8) {
       archNum += archMinor;
    }
    return archNum;
}

// Automatically generated 2025-01-17 from https://raw.githubusercontent.com/ROCm/clr/refs/heads/amd-staging/rocclr/device/device.cpp

int main() {
    struct target targets[124];

    strcpy(targets[0].id, "gfx801");
    targets[0].major = 8;
    targets[0].minor = 0;
    targets[0].step  = 1;
    strcpy(targets[1].id, "gfx801:xnack-");
    targets[1].major = 8;
    targets[1].minor = 0;
    targets[1].step  = 1;
    strcpy(targets[2].id, "gfx801:xnack+");
    targets[2].major = 8;
    targets[2].minor = 0;
    targets[2].step  = 1;
    strcpy(targets[3].id, "gfx802");
    targets[3].major = 8;
    targets[3].minor = 0;
    targets[3].step  = 2;
    strcpy(targets[4].id, "gfx803");
    targets[4].major = 8;
    targets[4].minor = 0;
    targets[4].step  = 3;
    strcpy(targets[5].id, "gfx805");
    targets[5].major = 8;
    targets[5].minor = 0;
    targets[5].step  = 5;
    strcpy(targets[6].id, "gfx810");
    targets[6].major = 8;
    targets[6].minor = 1;
    targets[6].step  = 0;
    strcpy(targets[7].id, "gfx810:xnack-");
    targets[7].major = 8;
    targets[7].minor = 1;
    targets[7].step  = 0;
    strcpy(targets[8].id, "gfx810:xnack+");
    targets[8].major = 8;
    targets[8].minor = 1;
    targets[8].step  = 0;
    strcpy(targets[9].id, "gfx900");
    targets[9].major = 9;
    targets[9].minor = 0;
    targets[9].step  = 0;
    strcpy(targets[10].id, "gfx900:xnack-");
    targets[10].major = 9;
    targets[10].minor = 0;
    targets[10].step  = 0;
    strcpy(targets[11].id, "gfx900:xnack+");
    targets[11].major = 9;
    targets[11].minor = 0;
    targets[11].step  = 0;
    strcpy(targets[12].id, "gfx902");
    targets[12].major = 9;
    targets[12].minor = 0;
    targets[12].step  = 2;
    strcpy(targets[13].id, "gfx902:xnack-");
    targets[13].major = 9;
    targets[13].minor = 0;
    targets[13].step  = 2;
    strcpy(targets[14].id, "gfx902:xnack+");
    targets[14].major = 9;
    targets[14].minor = 0;
    targets[14].step  = 2;
    strcpy(targets[15].id, "gfx904");
    targets[15].major = 9;
    targets[15].minor = 0;
    targets[15].step  = 4;
    strcpy(targets[16].id, "gfx904:xnack-");
    targets[16].major = 9;
    targets[16].minor = 0;
    targets[16].step  = 4;
    strcpy(targets[17].id, "gfx904:xnack+");
    targets[17].major = 9;
    targets[17].minor = 0;
    targets[17].step  = 4;
    strcpy(targets[18].id, "gfx906");
    targets[18].major = 9;
    targets[18].minor = 0;
    targets[18].step  = 6;
    strcpy(targets[19].id, "gfx906:sramecc-");
    targets[19].major = 9;
    targets[19].minor = 0;
    targets[19].step  = 6;
    strcpy(targets[20].id, "gfx906:sramecc+");
    targets[20].major = 9;
    targets[20].minor = 0;
    targets[20].step  = 6;
    strcpy(targets[21].id, "gfx906:xnack-");
    targets[21].major = 9;
    targets[21].minor = 0;
    targets[21].step  = 6;
    strcpy(targets[22].id, "gfx906:xnack+");
    targets[22].major = 9;
    targets[22].minor = 0;
    targets[22].step  = 6;
    strcpy(targets[23].id, "gfx906:sramecc-:xnack-");
    targets[23].major = 9;
    targets[23].minor = 0;
    targets[23].step  = 6;
    strcpy(targets[24].id, "gfx906:sramecc-:xnack+");
    targets[24].major = 9;
    targets[24].minor = 0;
    targets[24].step  = 6;
    strcpy(targets[25].id, "gfx906:sramecc+:xnack-");
    targets[25].major = 9;
    targets[25].minor = 0;
    targets[25].step  = 6;
    strcpy(targets[26].id, "gfx906:sramecc+:xnack+");
    targets[26].major = 9;
    targets[26].minor = 0;
    targets[26].step  = 6;
    strcpy(targets[27].id, "gfx908");
    targets[27].major = 9;
    targets[27].minor = 0;
    targets[27].step  = 8;
    strcpy(targets[28].id, "gfx908:sramecc-");
    targets[28].major = 9;
    targets[28].minor = 0;
    targets[28].step  = 8;
    strcpy(targets[29].id, "gfx908:sramecc+");
    targets[29].major = 9;
    targets[29].minor = 0;
    targets[29].step  = 8;
    strcpy(targets[30].id, "gfx908:xnack-");
    targets[30].major = 9;
    targets[30].minor = 0;
    targets[30].step  = 8;
    strcpy(targets[31].id, "gfx908:xnack+");
    targets[31].major = 9;
    targets[31].minor = 0;
    targets[31].step  = 8;
    strcpy(targets[32].id, "gfx908:sramecc-:xnack-");
    targets[32].major = 9;
    targets[32].minor = 0;
    targets[32].step  = 8;
    strcpy(targets[33].id, "gfx908:sramecc-:xnack+");
    targets[33].major = 9;
    targets[33].minor = 0;
    targets[33].step  = 8;
    strcpy(targets[34].id, "gfx908:sramecc+:xnack-");
    targets[34].major = 9;
    targets[34].minor = 0;
    targets[34].step  = 8;
    strcpy(targets[35].id, "gfx908:sramecc+:xnack+");
    targets[35].major = 9;
    targets[35].minor = 0;
    targets[35].step  = 8;
    strcpy(targets[36].id, "gfx909");
    targets[36].major = 9;
    targets[36].minor = 0;
    targets[36].step  = 2;
    strcpy(targets[37].id, "gfx909:xnack-");
    targets[37].major = 9;
    targets[37].minor = 0;
    targets[37].step  = 2;
    strcpy(targets[38].id, "gfx909:xnack+");
    targets[38].major = 9;
    targets[38].minor = 0;
    targets[38].step  = 2;
    strcpy(targets[39].id, "gfx90a");
    targets[39].major = 9;
    targets[39].minor = 0;
    targets[39].step  = 10;
    strcpy(targets[40].id, "gfx90a:sramecc-");
    targets[40].major = 9;
    targets[40].minor = 0;
    targets[40].step  = 10;
    strcpy(targets[41].id, "gfx90a:sramecc+");
    targets[41].major = 9;
    targets[41].minor = 0;
    targets[41].step  = 10;
    strcpy(targets[42].id, "gfx90a:xnack-");
    targets[42].major = 9;
    targets[42].minor = 0;
    targets[42].step  = 10;
    strcpy(targets[43].id, "gfx90a:xnack+");
    targets[43].major = 9;
    targets[43].minor = 0;
    targets[43].step  = 10;
    strcpy(targets[44].id, "gfx90a:sramecc-:xnack-");
    targets[44].major = 9;
    targets[44].minor = 0;
    targets[44].step  = 10;
    strcpy(targets[45].id, "gfx90a:sramecc-:xnack+");
    targets[45].major = 9;
    targets[45].minor = 0;
    targets[45].step  = 10;
    strcpy(targets[46].id, "gfx90a:sramecc+:xnack-");
    targets[46].major = 9;
    targets[46].minor = 0;
    targets[46].step  = 10;
    strcpy(targets[47].id, "gfx90a:sramecc+:xnack+");
    targets[47].major = 9;
    targets[47].minor = 0;
    targets[47].step  = 10;
    strcpy(targets[48].id, "gfx940");
    targets[48].major = 9;
    targets[48].minor = 4;
    targets[48].step  = 0;
    strcpy(targets[49].id, "gfx940:sramecc-");
    targets[49].major = 9;
    targets[49].minor = 4;
    targets[49].step  = 0;
    strcpy(targets[50].id, "gfx940:sramecc+");
    targets[50].major = 9;
    targets[50].minor = 4;
    targets[50].step  = 0;
    strcpy(targets[51].id, "gfx940:xnack-");
    targets[51].major = 9;
    targets[51].minor = 4;
    targets[51].step  = 0;
    strcpy(targets[52].id, "gfx940:xnack+");
    targets[52].major = 9;
    targets[52].minor = 4;
    targets[52].step  = 0;
    strcpy(targets[53].id, "gfx940:sramecc-:xnack-");
    targets[53].major = 9;
    targets[53].minor = 4;
    targets[53].step  = 0;
    strcpy(targets[54].id, "gfx940:sramecc-:xnack+");
    targets[54].major = 9;
    targets[54].minor = 4;
    targets[54].step  = 0;
    strcpy(targets[55].id, "gfx940:sramecc+:xnack-");
    targets[55].major = 9;
    targets[55].minor = 4;
    targets[55].step  = 0;
    strcpy(targets[56].id, "gfx940:sramecc+:xnack+");
    targets[56].major = 9;
    targets[56].minor = 4;
    targets[56].step  = 0;
    strcpy(targets[57].id, "gfx941");
    targets[57].major = 9;
    targets[57].minor = 4;
    targets[57].step  = 1;
    strcpy(targets[58].id, "gfx941:sramecc-");
    targets[58].major = 9;
    targets[58].minor = 4;
    targets[58].step  = 1;
    strcpy(targets[59].id, "gfx941:sramecc+");
    targets[59].major = 9;
    targets[59].minor = 4;
    targets[59].step  = 1;
    strcpy(targets[60].id, "gfx941:xnack-");
    targets[60].major = 9;
    targets[60].minor = 4;
    targets[60].step  = 1;
    strcpy(targets[61].id, "gfx941:xnack+");
    targets[61].major = 9;
    targets[61].minor = 4;
    targets[61].step  = 1;
    strcpy(targets[62].id, "gfx941:sramecc-:xnack-");
    targets[62].major = 9;
    targets[62].minor = 4;
    targets[62].step  = 1;
    strcpy(targets[63].id, "gfx941:sramecc-:xnack+");
    targets[63].major = 9;
    targets[63].minor = 4;
    targets[63].step  = 1;
    strcpy(targets[64].id, "gfx941:sramecc+:xnack-");
    targets[64].major = 9;
    targets[64].minor = 4;
    targets[64].step  = 1;
    strcpy(targets[65].id, "gfx941:sramecc+:xnack+");
    targets[65].major = 9;
    targets[65].minor = 4;
    targets[65].step  = 1;
    strcpy(targets[66].id, "gfx942");
    targets[66].major = 9;
    targets[66].minor = 4;
    targets[66].step  = 2;
    strcpy(targets[67].id, "gfx942:sramecc-");
    targets[67].major = 9;
    targets[67].minor = 4;
    targets[67].step  = 2;
    strcpy(targets[68].id, "gfx942:sramecc+");
    targets[68].major = 9;
    targets[68].minor = 4;
    targets[68].step  = 2;
    strcpy(targets[69].id, "gfx942:xnack-");
    targets[69].major = 9;
    targets[69].minor = 4;
    targets[69].step  = 2;
    strcpy(targets[70].id, "gfx942:xnack+");
    targets[70].major = 9;
    targets[70].minor = 4;
    targets[70].step  = 2;
    strcpy(targets[71].id, "gfx942:sramecc-:xnack-");
    targets[71].major = 9;
    targets[71].minor = 4;
    targets[71].step  = 2;
    strcpy(targets[72].id, "gfx942:sramecc-:xnack+");
    targets[72].major = 9;
    targets[72].minor = 4;
    targets[72].step  = 2;
    strcpy(targets[73].id, "gfx942:sramecc+:xnack-");
    targets[73].major = 9;
    targets[73].minor = 4;
    targets[73].step  = 2;
    strcpy(targets[74].id, "gfx942:sramecc+:xnack+");
    targets[74].major = 9;
    targets[74].minor = 4;
    targets[74].step  = 2;
    strcpy(targets[75].id, "gfx90c");
    targets[75].major = 9;
    targets[75].minor = 0;
    targets[75].step  = 12;
    strcpy(targets[76].id, "gfx90c:xnack-");
    targets[76].major = 9;
    targets[76].minor = 0;
    targets[76].step  = 12;
    strcpy(targets[77].id, "gfx90c:xnack+");
    targets[77].major = 9;
    targets[77].minor = 0;
    targets[77].step  = 12;
    strcpy(targets[78].id, "gfx9-generic");
    targets[78].major = 9;
    targets[78].minor = 0;
    targets[78].step  = 0;
    strcpy(targets[79].id, "gfx9-generic:xnack-");
    targets[79].major = 9;
    targets[79].minor = 0;
    targets[79].step  = 0;
    strcpy(targets[80].id, "gfx9-generic:xnack+");
    targets[80].major = 9;
    targets[80].minor = 0;
    targets[80].step  = 0;
    strcpy(targets[81].id, "gfx9-4-generic");
    targets[81].major = 9;
    targets[81].minor = 4;
    targets[81].step  = 0;
    strcpy(targets[82].id, "gfx9-4-generic:sramecc-");
    targets[82].major = 9;
    targets[82].minor = 4;
    targets[82].step  = 0;
    strcpy(targets[83].id, "gfx9-4-generic:sramecc+");
    targets[83].major = 9;
    targets[83].minor = 4;
    targets[83].step  = 0;
    strcpy(targets[84].id, "gfx9-4-generic:xnack-");
    targets[84].major = 9;
    targets[84].minor = 4;
    targets[84].step  = 0;
    strcpy(targets[85].id, "gfx9-4-generic:xnack+");
    targets[85].major = 9;
    targets[85].minor = 4;
    targets[85].step  = 0;
    strcpy(targets[86].id, "gfx9-4-generic:sramecc-:xnack-");
    targets[86].major = 9;
    targets[86].minor = 4;
    targets[86].step  = 0;
    strcpy(targets[87].id, "gfx9-4-generic:sramecc-:xnack+");
    targets[87].major = 9;
    targets[87].minor = 4;
    targets[87].step  = 0;
    strcpy(targets[88].id, "gfx9-4-generic:sramecc+:xnack-");
    targets[88].major = 9;
    targets[88].minor = 4;
    targets[88].step  = 0;
    strcpy(targets[89].id, "gfx9-4-generic:sramecc+:xnack+");
    targets[89].major = 9;
    targets[89].minor = 4;
    targets[89].step  = 0;
    strcpy(targets[90].id, "gfx1010");
    targets[90].major = 10;
    targets[90].minor = 1;
    targets[90].step  = 0;
    strcpy(targets[91].id, "gfx1010:xnack-");
    targets[91].major = 10;
    targets[91].minor = 1;
    targets[91].step  = 0;
    strcpy(targets[92].id, "gfx1010:xnack+");
    targets[92].major = 10;
    targets[92].minor = 1;
    targets[92].step  = 0;
    strcpy(targets[93].id, "gfx1011");
    targets[93].major = 10;
    targets[93].minor = 1;
    targets[93].step  = 1;
    strcpy(targets[94].id, "gfx1011:xnack-");
    targets[94].major = 10;
    targets[94].minor = 1;
    targets[94].step  = 1;
    strcpy(targets[95].id, "gfx1011:xnack+");
    targets[95].major = 10;
    targets[95].minor = 1;
    targets[95].step  = 1;
    strcpy(targets[96].id, "gfx1012");
    targets[96].major = 10;
    targets[96].minor = 1;
    targets[96].step  = 2;
    strcpy(targets[97].id, "gfx1012:xnack-");
    targets[97].major = 10;
    targets[97].minor = 1;
    targets[97].step  = 2;
    strcpy(targets[98].id, "gfx1012:xnack+");
    targets[98].major = 10;
    targets[98].minor = 1;
    targets[98].step  = 2;
    strcpy(targets[99].id, "gfx1013");
    targets[99].major = 10;
    targets[99].minor = 1;
    targets[99].step  = 3;
    strcpy(targets[100].id, "gfx1013:xnack-");
    targets[100].major = 10;
    targets[100].minor = 1;
    targets[100].step  = 3;
    strcpy(targets[101].id, "gfx1013:xnack+");
    targets[101].major = 10;
    targets[101].minor = 1;
    targets[101].step  = 3;
    strcpy(targets[102].id, "gfx10-1-generic");
    targets[102].major = 10;
    targets[102].minor = 1;
    targets[102].step  = 0;
    strcpy(targets[103].id, "gfx10-1-generic:xnack-");
    targets[103].major = 10;
    targets[103].minor = 1;
    targets[103].step  = 0;
    strcpy(targets[104].id, "gfx10-1-generic:xnack+");
    targets[104].major = 10;
    targets[104].minor = 1;
    targets[104].step  = 0;
    strcpy(targets[105].id, "gfx1030");
    targets[105].major = 10;
    targets[105].minor = 3;
    targets[105].step  = 0;
    strcpy(targets[106].id, "gfx1031");
    targets[106].major = 10;
    targets[106].minor = 3;
    targets[106].step  = 1;
    strcpy(targets[107].id, "gfx1032");
    targets[107].major = 10;
    targets[107].minor = 3;
    targets[107].step  = 2;
    strcpy(targets[108].id, "gfx1033");
    targets[108].major = 10;
    targets[108].minor = 3;
    targets[108].step  = 3;
    strcpy(targets[109].id, "gfx1034");
    targets[109].major = 10;
    targets[109].minor = 3;
    targets[109].step  = 4;
    strcpy(targets[110].id, "gfx1035");
    targets[110].major = 10;
    targets[110].minor = 3;
    targets[110].step  = 5;
    strcpy(targets[111].id, "gfx1036");
    targets[111].major = 10;
    targets[111].minor = 3;
    targets[111].step  = 6;
    strcpy(targets[112].id, "gfx10-3-generic");
    targets[112].major = 10;
    targets[112].minor = 3;
    targets[112].step  = 0;
    strcpy(targets[113].id, "gfx1100");
    targets[113].major = 11;
    targets[113].minor = 0;
    targets[113].step  = 0;
    strcpy(targets[114].id, "gfx1101");
    targets[114].major = 11;
    targets[114].minor = 0;
    targets[114].step  = 1;
    strcpy(targets[115].id, "gfx1102");
    targets[115].major = 11;
    targets[115].minor = 0;
    targets[115].step  = 2;
    strcpy(targets[116].id, "gfx1103");
    targets[116].major = 11;
    targets[116].minor = 0;
    targets[116].step  = 3;
    strcpy(targets[117].id, "gfx1150");
    targets[117].major = 11;
    targets[117].minor = 5;
    targets[117].step  = 0;
    strcpy(targets[118].id, "gfx1151");
    targets[118].major = 11;
    targets[118].minor = 5;
    targets[118].step  = 1;
    strcpy(targets[119].id, "gfx1152");
    targets[119].major = 11;
    targets[119].minor = 5;
    targets[119].step  = 2;
    strcpy(targets[120].id, "gfx11-generic");
    targets[120].major = 11;
    targets[120].minor = 0;
    targets[120].step  = 0;
    strcpy(targets[121].id, "gfx1200");
    targets[121].major = 12;
    targets[121].minor = 0;
    targets[121].step  = 0;
    strcpy(targets[122].id, "gfx1201");
    targets[122].major = 12;
    targets[122].minor = 0;
    targets[122].step  = 1;
    strcpy(targets[123].id, "gfx12-generic");
    targets[123].major = 12;
    targets[123].minor = 0;
    targets[123].step  = 0;

    int verReturned;
    int verActual;
    char * result;
    char pass[] = "OK  ";
    char fail[] = "FAIL";
    int total = 0;
    int good = 0;

    for (int i = 0; i < sizeof(targets) / sizeof(struct target); i++) {
        result = fail;
        total += 1;
        verActual = (targets[i].major % 10) * 0x10 * 0x10;
        verActual += (targets[i].major / 10) * 0x10 * 0x100;
        if (targets[i].major != 8) {
            verActual += targets[i].minor * 0x10;
            verActual += targets[i].step;
        }

        verReturned = ggml_cuda_parse_id(targets[i].id);
        if (verActual + GGML_CUDA_CC_OFFSET_AMD == verReturned) {
            result = pass;
            good += 1;
        } else {
            // gfx909 is mapped to 902
            if (verActual == 0x902 && (verReturned & 0xffff) == 0x909) {
                result = pass;
                good += 1;
            }
        }

        printf("%03d: %s: Actual: 0x%04x, Returned: 0x%04x, ID: %s\n",
            i, result, verActual, verReturned & 0xffff, targets[i].id);
    }
    printf("Total: %d  Passed: %d  Failed: %d\n", total, good, total - good);
    return total - good;
}
