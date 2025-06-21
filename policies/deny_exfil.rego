# SPDX-License-Identifier: Apache-2.0
package exfil

# Block suspicious data exfiltration commands

deny[msg] {
    re_match("(?i)curl\\s+https?://", input.text)
    msg := "exfiltration attempt blocked"
}

deny[msg] {
    re_match("(?i)scp\\s", input.text)
    msg := "exfiltration attempt blocked"
}
