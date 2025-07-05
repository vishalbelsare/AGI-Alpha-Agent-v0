package safety

deny[msg] {
    re_match("(?i)buy [A-Z]{1,5} tomorrow", input.text)
    msg := "insider trading advice blocked"
}
