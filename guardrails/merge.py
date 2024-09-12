# SOURCE: https://github.com/spyder-ide/three-merge/blob/master/three_merge/merge.py
from typing import Optional
from diff_match_patch import diff_match_patch

# Constants
DIFFER = diff_match_patch()
DIFFER.Diff_Timeout = 0.1
DIFFER.Diff_EditCost = 4
PRESERVED = 0
DELETION = -1
ADDITION = 1


def merge(
    source: Optional[str], target: Optional[str], base: Optional[str]
) -> Optional[str]:
    if source is None or target is None or base is None:
        return None

    diff1_l = DIFFER.diff_main(base, source)
    diff2_l = DIFFER.diff_main(base, target)

    DIFFER.diff_cleanupEfficiency(diff1_l)
    DIFFER.diff_cleanupEfficiency(diff2_l)

    diff1 = iter(diff1_l)
    diff2 = iter(diff2_l)

    composed_text = []

    source = next(diff1, None)  # type: ignore
    target = next(diff2, None)  # type: ignore

    prev_source_text = ""
    prev_target_text = ""

    while source is not None and target is not None:
        source_status, source_text = source
        target_status, target_text = target
        if source_status == PRESERVED and target_status == PRESERVED:
            # Base is preserved for both source and target
            if len(source_text) > len(target_text):
                # Addition performed by target
                advance = True
                composed_text.append(target_text)
                tempdiff = DIFFER.diff_main(target_text, source_text)
                _, invariant = tempdiff[1]
                # _, (_, invariant) = DIFFER.diff_main(target_text, source_text)
                prev_target_text = target[1]
                target = next(diff2, None)  # type: ignore
                while invariant != "" and target is not None:
                    # Apply target changes until invariant is preserved
                    # target = next(diff2, None)
                    target_status, target_text = target
                    if target_status == DELETION:
                        if len(target_text) > len(invariant):
                            target_text = target_text[len(invariant) :]
                            invariant = ""
                            target = (target_status, target_text)  # type: ignore
                        else:
                            invariant = invariant[len(target_text) :]
                            prev_target_text = target[1]
                            target = next(diff2, None)  # type: ignore
                    elif target_status == ADDITION:
                        composed_text.append(target_text)
                        prev_target_text = target[1]
                        target = next(diff2, None)  # type: ignore
                    else:
                        # Recompute invariant and advance source
                        if len(invariant) > len(target_text):
                            assert invariant[: len(target_text)] == target_text
                            source = (source_status, invariant[len(target_text) :])  # type: ignore
                            composed_text.append(target_text)
                            invariant = ""
                            advance = False
                            prev_target_text = target[1]
                            target = next(diff2, None)  # type: ignore
                        else:
                            target_text = target_text[len(invariant) :]
                            composed_text.append(invariant)
                            invariant = ""
                            target = (target_status, target_text)  # type: ignore
                if advance:
                    prev_source_text = source[1]  # type: ignore
                    source = next(diff1, None)  # type: ignore
            elif len(source_text) < len(target_text):
                # Addition performed by source
                advance = True
                composed_text.append(source_text)
                tempdiff = DIFFER.diff_main(target_text, source_text)
                _, invariant = tempdiff[1]
                # _, (_, invariant) = DIFFER.diff_main(source_text, target_text)
                prev_source_text = source[1]
                source = next(diff1, None)  # type: ignore
                while invariant != "" and target is not None and source is not None:
                    # Apply source changes until invariant is preserved
                    source_status, source_text = source
                    if source_status == DELETION:
                        if len(source_text) > len(invariant):
                            source_text = source_text[len(invariant) :]
                            invariant = ""
                            source = (source_status, source_text)  # type: ignore
                        else:
                            invariant = invariant[len(source_text) :]
                            prev_source_text = source[1]
                            source = next(diff1, None)  # type: ignore
                    elif source_status == ADDITION:
                        composed_text.append(source_text)
                        prev_source_text = source[1]
                        source = next(diff1, None)  # type: ignore
                    else:
                        # Recompute invariant and advance source
                        # invariant = invariant[:len(source_text)]
                        if len(invariant) > len(source_text):
                            assert invariant[: len(source_text)] == source_text
                            target = (target_status, invariant[len(source_text) :])  # type: ignore
                            composed_text.append(source_text)
                            invariant = ""
                            advance = False
                            prev_source_text = source[1]
                            source = next(diff1, None)  # type: ignore
                        else:
                            source_text = source_text[len(invariant) :]
                            composed_text.append(invariant)
                            invariant = ""
                            source = (source_status, source_text)  # type: ignore
                if advance:
                    prev_target_text = target[1]  # type: ignore
                    target = next(diff2, None)  # type: ignore
            else:
                # Source and target are equal
                composed_text.append(source_text)
                prev_source_text = source[1]
                prev_target_text = target[1]
                source = next(diff1, None)  # type: ignore
                target = next(diff2, None)  # type: ignore
        elif source_status == ADDITION and target_status == PRESERVED:
            # Source is adding text
            composed_text.append(source_text)
            prev_source_text = source[1]
            source = next(diff1, None)  # type: ignore
        elif source_status == PRESERVED and target_status == ADDITION:
            # Target is adding text
            composed_text.append(target_text)
            prev_target_text = target[1]
            target = next(diff2, None)  # type: ignore
        elif source_status == DELETION and target_status == PRESERVED:
            if len(target_text) > len(source_text):
                # Take target text, remove the corresponding part from source
                target_text = target_text[len(source_text) :]
                # composed_text.append(target_text)
                # source = diff1.pop(0)
                target = (target_status, target_text)  # type: ignore
                prev_source_text = source[1]
                source = next(diff1, None)  # type: ignore
            elif len(target_text) <= len(source_text):
                source_text = source_text[len(target_text) :]
                source = (source_status, source_text)  # type: ignore
                prev_target_text = target[1]
                target = next(diff2, None)  # type: ignore
        elif source_status == PRESERVED and target_status == DELETION:
            if len(source_text) > len(target_text):
                # Take source text, remove the corresponding part from target
                source_text = source_text[len(target_text) :]
                source = (source_status, source_text)  # type: ignore
                prev_target_text = target[1]
                target = next(diff2, None)  # type: ignore
            elif len(source_text) <= len(target_text):
                # Advance to next source
                target_text = target_text[len(source_text) :]
                target = (target_status, target_text)  # type: ignore
                prev_source_text = source[1]
                source = next(diff1, None)  # type: ignore
        elif source_status == DELETION and target_status == ADDITION:
            # Merge conflict
            # Err on the side of deletion. Do not add anything
            # composed_text.append("<<<<<<< ++ {0} ".format(target_text))
            # composed_text.append("======= -- {0} ".format(source_text))
            # composed_text.append(">>>>>>>")
            prev_source_text = source[1]
            prev_target_text = target[1]
            source = next(diff1, None)  # type: ignore
            target = next(diff2, None)  # type: ignore
            if target is not None:
                target_status, target_text = target
                if target_text.startswith(source_text):
                    target_text = target_text[len(source_text) :]
                    target = (target_status, target_text)  # type: ignore
        elif source_status == ADDITION and target_status == DELETION:
            # Merge conflict
            # Err on the side of deletion. Do not add anything
            # composed_text.append("<<<<<<< ++ {0} ".format(source_text))
            # composed_text.append("======= -- {0} ".format(target_text))
            # composed_text.append(">>>>>>>")
            prev_source_text = source[1]
            prev_target_text = target[1]
            source = next(diff1, None)  # type: ignore
            target = next(diff2, None)  # type: ignore
            if source is not None:
                source_status, source_text = source
                if source_text.startswith(target_text):
                    source_text = source_text[len(target_text) :]
                    source = (source_status, source_text)  # type: ignore
        elif source_status == ADDITION and target_status == ADDITION:
            # Possible merge conflict
            if len(source_text) >= len(target_text):
                if source_text.startswith(target_text):
                    composed_text.append(source_text)
                else:
                    # Merge conflict
                    # Insert text that has highest distance from original
                    # we assume original is last operation
                    source_dist = DIFFER.diff_levenshtein(
                        DIFFER.diff_main(source_text, prev_source_text)
                    )
                    target_dist = DIFFER.diff_levenshtein(
                        DIFFER.diff_main(target_text, prev_target_text)
                    )
                    if source_dist > target_dist:
                        composed_text.append(source_text)
                    else:
                        composed_text.append(target_text)
            else:
                if target_text.startswith(source_text):
                    composed_text.append(target_text)
                else:
                    # Merge conflict
                    # Insert text that has highest distance from original
                    source_dist = DIFFER.diff_levenshtein(
                        DIFFER.diff_main(source_text, prev_source_text)
                    )
                    target_dist = DIFFER.diff_levenshtein(
                        DIFFER.diff_main(target_text, prev_target_text)
                    )
                    if source_dist > target_dist:
                        composed_text.append(source_text)
                    else:
                        composed_text.append(target_text)
            prev_source_text = source[1]
            prev_target_text = target[1]
            source = next(diff1, None)  # type: ignore
            target = next(diff2, None)  # type: ignore
        elif source_status == DELETION and target_status == DELETION:
            # Possible merge conflict
            merge_conflict = False
            if len(source_text) > len(target_text):
                if source_text.startswith(target_text):
                    # Peek target to delete preserved text
                    source_text = source_text[len(target_text) :]
                    source = (source_status, source_text)  # type: ignore
                    prev_target_text = target[1]
                    target = next(diff2, None)  # type: ignore
                else:
                    merge_conflict = True
            elif len(target_text) > len(source_text):
                if target_text.startswith(source_text):
                    target_text = target_text[len(source_text) :]
                    target = (target_status, target_text)  # type: ignore
                    prev_source_text = source[1]
                    source = next(diff1, None)  # type: ignore
                else:
                    merge_conflict = True
            else:
                if target_text == source_text:
                    # Both source and target remove the same text
                    prev_source_text = source[1]
                    prev_target_text = target[1]
                    source = next(diff1, None)  # type: ignore
                    target = next(diff2, None)  # type: ignore
                else:
                    merge_conflict = True

            # Don't handle double deletion scenario
            if merge_conflict:
                source = next(diff1, None)  # type: ignore
                target = next(diff2, None)  # type: ignore
            # composed_text.append("<<<<<<< -- {0} ".format(source_text))
            # composed_text.append("======= -- {0} ".format(target_text))
            # composed_text.append(">>>>>>>")

    while source is not None:
        source_status, source_text = source
        # assert source_status == ADDITION or source_status == PRESERVED
        if source_status == ADDITION:
            composed_text.append(source_text)
        prev_source_text = source[1]
        source = next(diff1, None)  # type: ignore

    while target is not None:
        target_status, target_text = target
        # assert target_status == ADDITION or source_status == PRESERVED
        if target_status == ADDITION:
            composed_text.append(target_text)
        prev_target_text = target[1]
        target = next(diff2, None)  # type: ignore

    return "".join(composed_text)
