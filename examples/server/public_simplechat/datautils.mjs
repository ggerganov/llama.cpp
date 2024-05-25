//@ts-check
// Helpers to work with different data types
// by Humans for All
//


/**
 * Simple minded logic to help remove repeating garbage at end of the string.
 * @param {string} sIn
 * @param {number} maxSubL
 * @param {number} maxMatchLenThreshold
 */
export function trim_repeat_garbage_at_end(sIn, maxSubL=10, maxMatchLenThreshold=40) {
    let rCnt = [0];
    let maxMatchLen = maxSubL;
    let iMML = -1;
    for(let subL=1; subL < maxSubL; subL++) {
        rCnt.push(0);
        let i;
        let refS = sIn.substring(sIn.length-subL, sIn.length);
        for(i=sIn.length; i > 0; i -= subL) {
            let curS = sIn.substring(i-subL, i);
            if (refS != curS) {
                let curMatchLen = rCnt[subL]*subL;
                if (maxMatchLen < curMatchLen) {
                    maxMatchLen = curMatchLen;
                    iMML = subL;
                }
                break;
            }
            rCnt[subL] += 1;
        }
    }
    console.log("DBUG:DU:TrimRepeatGarbage:", rCnt);
    if ((iMML == -1) || (maxMatchLen < maxMatchLenThreshold)) {
        return {trimmed: false, data: sIn};
    }
    let iEnd = sIn.length - maxMatchLen;
    return { trimmed: true, data: sIn.substring(0, iEnd) };
}


/**
 * Simple minded logic to help remove repeating garbage at end of the string, till it cant.
 * @param {string} sIn
 * @param {number} maxSubL
 * @param {number | undefined} [maxMatchLenThreshold]
 */
export function trim_repeat_garbage_at_end_loop(sIn, maxSubL, maxMatchLenThreshold) {
    let sCur = sIn;
    while(true) {
        let got = trim_repeat_garbage_at_end(sCur, maxSubL, maxMatchLenThreshold);
        if (got.trimmed != true) {
            return got.data;
        }
        sCur = got.data;
    }
}
