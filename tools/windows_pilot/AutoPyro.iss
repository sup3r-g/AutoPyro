; Compile with ISCC /DPayloadDir=<unpacked standalone kit> /O<output> AutoPyro.iss
#ifndef PayloadDir
  #error PayloadDir must point to the extracted RC3 standalone ZIP
#endif

[Setup]
AppId=AutoPyro-0.1.1rc3
AppName=AutoPyro
AppVersion=0.1.1rc3
DefaultDirName={localappdata}\Programs\AutoPyro\0.1.1rc3
DefaultGroupName=AutoPyro 0.1.1rc3
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputBaseFilename=AutoPyro-0.1.1rc3-Setup-x64
DisableWelcomePage=yes
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
WizardStyle=modern
Compression=zip
SolidCompression=no
SetupLogging=yes
ExtraDiskSpaceRequired=2200000000

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Files]
Source: "{#PayloadDir}\runtime\*"; DestDir: "{app}\runtime"; Flags: recursesubdirs ignoreversion
Source: "{#PayloadDir}\control\*"; DestDir: "{app}\control"; Flags: recursesubdirs ignoreversion
Source: "{#PayloadDir}\AutoPyro-0.1.1rc3-windows-x64-py312.zip"; DestDir: "{app}"; Flags: nocompression ignoreversion
Source: "check-installed.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_and_check.py"; DestDir: "{app}"; Flags: ignoreversion; AfterInstall: InstallAndVerify

[Icons]
Name: "{group}\AutoPyro"; Filename: "{app}\env\Scripts\autopyro.exe"; WorkingDir: "{app}"
Name: "{group}\Контрольная проверка"; Filename: "{app}\check-installed.cmd"; WorkingDir: "{app}"
Name: "{group}\Протоколы проверок"; Filename: "{app}\checks"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\env"

[Messages]
FinishedLabel=Установка AutoPyro и контрольная проверка завершены. Приложение и протоколы проверок доступны в меню «Пуск», в папке «AutoPyro 0.1.1rc3».

[Code]
procedure InstallAndVerify;
var
  Code: Integer;
begin
  WizardForm.StatusLabel.Caption := 'Установка пакетов и проверка отчёта и восстановления…';
  if not Exec(ExpandConstant('{app}\runtime\python.exe'),
    '-I -X utf8 "' + ExpandConstant('{app}\install_and_check.py') + '" setup',
    ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code) then
    RaiseException('Не удалось запустить комплектный Python: ' + SysErrorMessage(Code));
  if Code <> 0 then
    RaiseException('Ошибка установки или контрольной проверки. Код ' + IntToStr(Code) +
      '. Журнал: ' + ExpandConstant('{app}\checks'));
end;
