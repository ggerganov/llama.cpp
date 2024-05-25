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
    console.debug("DBUG:DU:TrimRepeatGarbage:", rCnt);
    if ((iMML == -1) || (maxMatchLen < maxMatchLenThreshold)) {
        return {trimmed: false, data: sIn};
    }
    console.debug("DBUG:TrimRepeatGarbage:TrimmedCharLen:", maxMatchLen);
    let iEnd = sIn.length - maxMatchLen;
    return { trimmed: true, data: sIn.substring(0, iEnd) };
}


/**
 * Simple minded logic to help remove repeating garbage at end of the string, till it cant.
 * If its not able to trim, then it will try to skip a char at end and then trim, a few times.
 * @param {string} sIn
 * @param {number} maxSubL
 * @param {number | undefined} [maxMatchLenThreshold]
 */
export function trim_repeat_garbage_at_end_loop(sIn, maxSubL, maxMatchLenThreshold, skipMax=16) {
    let sCur = sIn;
    let sSaved = "";
    let iTry = 0;
    while(true) {
        let got = trim_repeat_garbage_at_end(sCur, maxSubL, maxMatchLenThreshold);
        if (got.trimmed != true) {
            if (iTry == 0) {
                sSaved = got.data;
            }
            iTry += 1;
            if (iTry >= skipMax) {
                return sSaved;
            }
            got.data = got.data.substring(0,got.data.length-1);
        } else {
            iTry = 0;
        }
        sCur = got.data;
    }
}


/**
 * A simple minded try trim garbage at end using histogram characteristics
 *
 * Allow the garbage to contain upto maxUniq chars, but at the same time ensure that
 * a given type of char ie numerals or alphabets or other types dont cross the specified
 * maxType limit. This allows intermixed text garbage to be identified and trimmed.
 *
 * ALERT: This is not perfect and only provides a rough garbage identification logic.
 * Also it currently only differentiates between character classes wrt english.
 *
 * @param {string} sIn
 * @param {number} maxType
 * @param {number} maxUniq
 * @param {number} maxMatchLenThreshold
 */
export function trim_hist_garbage_at_end(sIn, maxType, maxUniq, maxMatchLenThreshold) {
    if (sIn.length < maxMatchLenThreshold) {
        return { trimmed: false, data: sIn };
    }
    let iAlp = 0;
    let iNum = 0;
    let iOth = 0;
    // Learn
    let hist = {};
    let iUniq = 0;
    for(let i=0; i<maxMatchLenThreshold; i++) {
        let c = sIn[sIn.length-1-i];
        if (c in hist) {
            hist[c] += 1;
        } else {
            if(c.match(/[0-9]/) != null) {
                iNum += 1;
            } else if(c.match(/[A-Za-z]/) != null) {
                iAlp += 1;
            } else {
                iOth += 1;
            }
            iUniq += 1;
            if (iUniq >= maxUniq) {
                break;
            }
            hist[c] = 1;
        }
    }
    console.debug("DBUG:TrimHistGarbage:", hist);
    if ((iAlp > maxType) || (iNum > maxType) || (iOth > maxType)) {
        return { trimmed: false, data: sIn };
    }
    // Catch and Trim
    for(let i=0; i < sIn.length; i++) {
        let c = sIn[sIn.length-1-i];
        if (!(c in hist)) {
            if (i < maxMatchLenThreshold) {
                return { trimmed: false, data: sIn };
            }
            console.debug("DBUG:TrimHistGarbage:TrimmedCharLen:", i);
            return { trimmed: true, data: sIn.substring(0, sIn.length-i+1) };
        }
    }
    console.debug("DBUG:TrimHistGarbage:Trimmed fully");
    return { trimmed: true, data: "" };
}

/**
 * Keep trimming repeatedly using hist_garbage logic, till you no longer can
 * @param {any} sIn
 * @param {number} maxType
 * @param {number} maxUniq
 * @param {number} maxMatchLenThreshold
 */
export function trim_hist_garbage_at_end_loop(sIn, maxType, maxUniq, maxMatchLenThreshold) {
    let sCur = sIn;
    while (true) {
        let got = trim_hist_garbage_at_end(sCur, maxType, maxUniq, maxMatchLenThreshold);
        if (!got.trimmed) {
            return got.data;
        }
        sCur = got.data;
    }
}
