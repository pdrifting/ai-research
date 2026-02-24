@echo off
set count=0

:loop
if %count% leq 100 (
   echo Current Run: %count%
   ionai.exe
   set /a count=%count%+1
   goto loop    
)