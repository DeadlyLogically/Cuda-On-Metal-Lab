#pragma once

int bridge_call(char **reply, const char *format, ...)
    __attribute__((format(printf, 2, 3)));
int bridge_number(unsigned long long *number, const char *format, ...)
    __attribute__((format(printf, 2, 3)));
char *hex_encode(const void *bytes, size_t count);
int hex_decode(const char *hex, void *bytes, size_t count);
