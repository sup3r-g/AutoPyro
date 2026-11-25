; ISCC /DPayloadDir=<prepared standalone folder> /O<output> AutoPyro-Nuitka.iss
#ifndef PayloadDir
  #error Define PayloadDir for the standalone build with control/installed-files.json
#endif

[Setup]
AppId=AutoPyro-Nuitka-0.1.1rc3
AppName=AutoPyro (Nuitka)
AppVersion=0.1.1rc3
DefaultDirName={localappdata}\Programs\AutoPyro\0.1.1rc3-Nuitka
DefaultGroupName=AutoPyro 0.1.1rc3 (Nuitka)
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputBaseFilename=AutoPyro-0.1.1rc3-Nuitka-Setup-x64
DisableWelcomePage=yes
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
SetupLogging=yes
UninstallDisplayIcon={app}\AutoPyro.exe

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Excludes: "checks\*"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\AutoPyro"; Filename: "{app}\AutoPyro.exe"; WorkingDir: "{app}"
Name: "{group}\Контрольная проверка"; Filename: "{app}\check-installed.cmd"; WorkingDir: "{app}"
Name: "{group}\Протоколы проверок"; Filename: "{app}\checks"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
begin
  if CurStep = ssPostInstall then begin
    WizardForm.StatusLabel.Caption := 'Проверка файлов, отчёта и восстановления…';
    if not Exec(ExpandConstant('{app}\AutoPyro.exe'), '--check-installation',
      ExpandConstant('{app}'), SW_SHOWNORMAL, ewWaitUntilTerminated, Code) then
      RaiseException('Не удалось запустить контрольную проверку: ' + SysErrorMessage(Code));
    if Code <> 0 then
      RaiseException('Ошибка контрольной проверки. Код ' + IntToStr(Code) +
        '. Протоколы: ' + ExpandConstant('{app}\checks'));
  end;
end;
