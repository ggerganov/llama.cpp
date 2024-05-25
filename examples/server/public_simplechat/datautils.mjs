//@ts-check
// Helpers to work with different data types
// by Humans for All
//


/**
 * Simple minded logic to help remove repeating garbage at end of the string.
 * TODO: Initial skeleton
 * @param {string} sIn
 */
export function trim_repeat_garbage_at_end(sIn, maxSubL=10) {
    let rCnt = [0];
    const MaxMatchLenThreshold = maxSubL*4;
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
    if ((iMML == -1) || (maxMatchLen < MaxMatchLenThreshold)) {
        return sIn;
    }
    let iEnd = sIn.length - maxMatchLen + iMML;
    return sIn.substring(0,iEnd)
}
