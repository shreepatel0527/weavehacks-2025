#!/usr/bin/env node

const { spawn } = require('child_process');

console.log('Testing Claude command availability...\n');

const commands = [
  { name: 'claude CLI', cmd: 'claude', args: ['--version'] },
  { name: 'claude-flow via npx', cmd: 'npx', args: ['claude-flow@alpha', '--version'] },
  { name: 'claude-code via npx', cmd: 'npx', args: ['@anthropic-ai/claude-code', '--version'] },
  { name: 'which claude', cmd: 'which', args: ['claude'] },
];

async function testCommand(name, cmd, args) {
  return new Promise((resolve) => {
    console.log(`Testing ${name}...`);
    const proc = spawn(cmd, args, { shell: true });
    
    let output = '';
    let error = '';
    
    proc.stdout.on('data', (data) => { output += data.toString(); });
    proc.stderr.on('data', (data) => { error += data.toString(); });
    
    proc.on('close', (code) => {
      if (code === 0) {
        console.log(`✅ ${name}: SUCCESS`);
        if (output) console.log(`   Output: ${output.trim()}`);
      } else {
        console.log(`❌ ${name}: FAILED (code ${code})`);
        if (error) console.log(`   Error: ${error.trim()}`);
      }
      console.log('');
      resolve();
    });
    
    proc.on('error', (err) => {
      console.log(`❌ ${name}: ERROR - ${err.message}\n`);
      resolve();
    });
  });
}

async function runTests() {
  for (const { name, cmd, args } of commands) {
    await testCommand(name, cmd, args);
  }
  
  console.log('\nTesting a simple prompt with available commands...\n');
  
  // Test actual usage
  const testPrompt = 'Say "Hello from Claude" and nothing else';
  
  // Try claude CLI first
  const claudeTest = spawn('claude', ['-p', testPrompt], { shell: true });
  let claudeWorked = false;
  
  claudeTest.stdout.on('data', (data) => {
    console.log('Claude CLI output:', data.toString());
    claudeWorked = true;
  });
  
  claudeTest.on('close', async (code) => {
    if (!claudeWorked) {
      console.log('Claude CLI not available, trying npx...\n');
      
      // Try npx claude-flow
      const npxTest = spawn('npx', ['claude-flow@alpha', '-p', testPrompt], { 
        shell: true,
        timeout: 30000 
      });
      
      npxTest.stdout.on('data', (data) => {
        console.log('NPX claude-flow output:', data.toString());
      });
      
      npxTest.stderr.on('data', (data) => {
        console.log('NPX error:', data.toString());
      });
    }
  });
}

runTests();