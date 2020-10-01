<?php
$input = $_POST['inp'];
$command = 'echo '.$input.' | python results.py';
$out = shell_exec($command);
echo $out;
?>