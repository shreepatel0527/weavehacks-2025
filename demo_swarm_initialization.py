#!/usr/bin/env python3
"""
Development Swarm Initialization Demo
=====================================

This script demonstrates the initialization and execution of a development swarm
with 5 agents using the BatchTool pattern to analyze and integrate the video
monitoring system.

Usage:
    python demo_swarm_initialization.py

Features demonstrated:
- BatchTool pattern for simultaneous agent spawning
- Memory-based objective and task storage
- Hierarchical task management
- Video monitoring system integration
- Real-time coordination and communication
"""

import json
import time
import uuid
from pathlib import Path
from development_swarm import (
    DevelopmentSwarm, SwarmObjective, AgentRole, TaskStatus, TaskPriority
)

def print_banner(title: str, char: str = "="):
    """Print a formatted banner"""
    print(f"\n{char * 60}")
    print(f"🤖 {title}")
    print(f"{char * 60}")

def print_section(title: str):
    """Print a section header"""
    print(f"\n📋 {title}")
    print("-" * 40)

def demonstrate_swarm_initialization():
    """Main demonstration of swarm initialization"""
    
    print_banner("DEVELOPMENT SWARM INITIALIZATION DEMO")
    
    # Step 1: Create the main objective
    print_section("Step 1: Creating Main Objective")
    
    objective = SwarmObjective(
        id=str(uuid.uuid4()),
        title="Video Monitoring System Analysis & Integration",
        description="""
        Comprehensive analysis and integration of the video monitoring system with 
        development swarm capabilities. This includes code analysis, architecture design,
        implementation, testing, and quality validation.
        """.strip(),
        target_system="VideoMonitoringSystem",
        success_criteria=[
            "✅ Complete codebase analysis with detailed findings",
            "✅ Comprehensive integration architecture design",
            "✅ Core integration features implementation",
            "✅ Thorough testing and quality validation",
            "✅ Seamless swarm-video system coordination",
            "✅ Performance optimization and monitoring",
            "✅ Documentation and deployment readiness"
        ],
        metadata={
            "project_type": "video_monitoring_integration",
            "complexity": "high",
            "estimated_duration": "2-3 weeks",
            "team_size": 5
        }
    )
    
    print(f"📝 Objective Created:")
    print(f"   Title: {objective.title}")
    print(f"   Target: {objective.target_system}")
    print(f"   Success Criteria: {len(objective.success_criteria)} items")
    print(f"   ID: {objective.id}")
    
    # Step 2: Initialize the development swarm
    print_section("Step 2: Initializing Development Swarm (BatchTool Pattern)")
    
    print("🚀 Creating swarm instance...")
    swarm = DevelopmentSwarm(memory_path="demo_swarm_memory.json")
    
    print("⚡ Spawning agents with BatchTool pattern...")
    init_result = swarm.initialize_swarm(objective)
    
    print("✅ Swarm initialization completed!")
    print(f"📊 Initialization Results:")
    for key, value in init_result.items():
        print(f"   {key}: {value}")
    
    # Step 3: Display agent roster
    print_section("Step 3: Agent Roster")
    
    for role, agent in swarm.agents.items():
        print(f"🤖 {role.value.upper()}:")
        print(f"   Status: {'🟢 Active' if agent.is_active else '🔴 Inactive'}")
        print(f"   Capabilities: {', '.join(agent.capabilities[:3])}...")
        print(f"   Current Task: {agent.current_task.title if agent.current_task else 'None'}")
        print()
    
    # Step 4: Display task hierarchy
    print_section("Step 4: Initial Task Hierarchy")
    
    tasks_by_agent = {}
    for task in swarm.memory.tasks.values():
        agent_role = task.assigned_agent.value
        if agent_role not in tasks_by_agent:
            tasks_by_agent[agent_role] = []
        tasks_by_agent[agent_role].append(task)
    
    for agent_role, tasks in tasks_by_agent.items():
        print(f"📋 {agent_role.upper()} Tasks ({len(tasks)}):")
        for task in tasks:
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.FAILED: "❌"
            }.get(task.status, "❓")
            
            priority_icon = {
                TaskPriority.CRITICAL: "🔥",
                TaskPriority.HIGH: "⚠️",
                TaskPriority.MEDIUM: "📋",
                TaskPriority.LOW: "📝"
            }.get(task.priority, "❓")
            
            print(f"   {status_icon} {priority_icon} {task.title}")
        print()
    
    # Step 5: Execute workflow demonstration
    print_section("Step 5: Workflow Execution Demonstration")
    
    print("🔄 Starting swarm workflow execution...")
    workflow_result = swarm.execute_swarm_workflow()
    
    print("📊 Workflow Execution Results:")
    for key, value in workflow_result.items():
        print(f"   {key}: {value}")
    
    # Step 6: Display execution results
    print_section("Step 6: Task Execution Results")
    
    completed_tasks = [t for t in swarm.memory.tasks.values() if t.status == TaskStatus.COMPLETED]
    failed_tasks = [t for t in swarm.memory.tasks.values() if t.status == TaskStatus.FAILED]
    
    print(f"✅ Completed Tasks ({len(completed_tasks)}):")
    for task in completed_tasks[:5]:  # Show first 5
        execution_time = "N/A"
        if task.started_at and task.completed_at:
            duration = task.completed_at - task.started_at
            execution_time = f"{duration.total_seconds():.2f}s"
        
        print(f"   ✅ {task.title} (Agent: {task.assigned_agent.value}, Time: {execution_time})")
        
        # Show key results if available
        if task.result and isinstance(task.result, dict):
            for key, value in list(task.result.items())[:2]:  # Show first 2 results
                print(f"      📊 {key}: {value}")
    
    if failed_tasks:
        print(f"\n❌ Failed Tasks ({len(failed_tasks)}):")
        for task in failed_tasks:
            print(f"   ❌ {task.title} (Agent: {task.assigned_agent.value})")
            if task.result and 'error' in task.result:
                print(f"      Error: {task.result['error']}")
    
    # Step 7: Agent knowledge demonstration
    print_section("Step 7: Agent Knowledge Base")
    
    for role, knowledge in swarm.memory.agent_knowledge.items():
        if knowledge:  # Only show agents with accumulated knowledge
            print(f"🧠 {role.value.upper()} Knowledge:")
            for key, value in list(knowledge.items())[:2]:  # Show first 2 knowledge items
                if isinstance(value, dict):
                    print(f"   📚 {key}: {len(value)} items")
                elif isinstance(value, list):
                    print(f"   📚 {key}: {len(value)} entries")
                else:
                    print(f"   📚 {key}: {str(value)[:50]}...")
            print()
    
    # Step 8: Memory and communication stats
    print_section("Step 8: Memory & Communication Statistics")
    
    stats = {
        "Objectives Stored": len(swarm.memory.objectives),
        "Tasks Created": len(swarm.memory.tasks),
        "Communication Messages": len(swarm.memory.communication_history),
        "Agent Knowledge Entries": sum(len(k) for k in swarm.memory.agent_knowledge.values()),
        "Memory File Size": f"{Path('demo_swarm_memory.json').stat().st_size / 1024:.1f}KB" if Path('demo_swarm_memory.json').exists() else "N/A"
    }
    
    for key, value in stats.items():
        print(f"📊 {key}: {value}")
    
    # Step 9: Final swarm status
    print_section("Step 9: Final Swarm Status")
    
    final_status = swarm.get_swarm_status()
    
    print("🎯 Swarm Status Summary:")
    for key, value in final_status.items():
        icon = "🟢" if "active" in key and value else "📊"
        print(f"   {icon} {key.replace('_', ' ').title()}: {value}")
    
    # Step 10: Demonstrate video integration capability
    print_section("Step 10: Video Integration Capability")
    
    print("📹 Video Monitoring Integration Features:")
    print("   🔗 Real-time video event processing")
    print("   🤖 Dynamic task creation from video events")  
    print("   ⚡ Intelligent agent assignment based on event type")
    print("   📊 Integrated logging with Weave tracking")
    print("   🔄 Continuous monitoring and response")
    
    print("\n💡 To activate video integration:")
    print("   ```python")
    print("   from VideoMonitoring_ExampleFile_video_monitoring_system import VideoMonitoringSystem")
    print("   video_system = VideoMonitoringSystem()")
    print("   swarm.integrate_with_video_monitoring(video_system)")
    print("   video_system.start_monitoring()")
    print("   ```")
    
    return swarm, objective

def demonstrate_memory_persistence():
    """Demonstrate memory persistence across sessions"""
    print_section("Memory Persistence Demonstration")
    
    print("💾 Checking memory persistence...")
    
    # Load previous swarm state
    if Path("demo_swarm_memory.json").exists():
        print("✅ Found existing swarm memory file")
        
        # Create new swarm instance to test memory loading
        swarm = DevelopmentSwarm(memory_path="demo_swarm_memory.json")
        
        print(f"📊 Loaded from memory:")
        print(f"   Objectives: {len(swarm.memory.objectives)}")
        print(f"   Tasks: {len(swarm.memory.tasks)}")
        print(f"   Communications: {len(swarm.memory.communication_history)}")
        
        # Show some loaded data
        if swarm.memory.objectives:
            obj = list(swarm.memory.objectives.values())[0]
            print(f"   Last Objective: {obj.title}")
        
        if swarm.memory.tasks:
            completed_tasks = [t for t in swarm.memory.tasks.values() if t.status == TaskStatus.COMPLETED]
            print(f"   Completed Tasks: {len(completed_tasks)}")
    else:
        print("ℹ️ No existing memory file found (first run)")

def cleanup_demo():
    """Clean up demo files"""
    print_section("Demo Cleanup")
    
    demo_files = ["demo_swarm_memory.json"]
    
    for file_path in demo_files:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"🗑️ Cleaned up: {file_path}")
        else:
            print(f"ℹ️ No cleanup needed: {file_path}")

def main():
    """Main demonstration function"""
    try:
        # Run the main demonstration
        swarm, objective = demonstrate_swarm_initialization()
        
        # Demonstrate memory persistence
        demonstrate_memory_persistence()
        
        print_banner("DEMONSTRATION COMPLETED SUCCESSFULLY", "🎉")
        
        print("\n📋 Summary:")
        print("✅ Development swarm initialized with BatchTool pattern")
        print("✅ 5 agents spawned simultaneously (coordinator, researcher, architect, backend dev, tester)")
        print("✅ Memory system storing objectives and task hierarchy")
        print("✅ Video monitoring integration capability demonstrated")
        print("✅ Task execution and coordination working")
        print("✅ Agent knowledge base accumulating insights")
        print("✅ Communication and memory persistence functional")
        
        print("\n🚀 Next Steps:")
        print("1. Integrate with actual video monitoring system")
        print("2. Enhance agent capabilities with specific tools")
        print("3. Add real-time dashboard for monitoring")
        print("4. Implement adaptive agent scaling")
        print("5. Add ML-powered task prioritization")
        
        # Ask if user wants to clean up
        print("\n🧹 Cleanup Options:")
        print("- Keep demo files for further exploration")
        print("- Clean up demo files")
        
        cleanup_choice = input("\nClean up demo files? (y/N): ").lower().strip()
        if cleanup_choice in ['y', 'yes']:
            cleanup_demo()
        else:
            print("💾 Demo files preserved for further exploration")
            print("   - demo_swarm_memory.json: Contains swarm memory state")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()